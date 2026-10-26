"""Agent 主循环：人格设定 + 个人知识检索（RAG）+ 工具调用。

每次回答的流程：
  1. 用当前问题在个人知识库里检索相关片段；
  2. 组装 system prompt（人设 + 当前时间 + 最近长期记忆）；
  3. 把检索片段拼进 user 消息，交给本地模型；
  4. 若模型返回 tool_calls，则执行工具并回填结果，循环（有轮数上限）。
"""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path
from typing import Any

from .memory import KnowledgeBase, LongTermMemory
from .ollama_client import OllamaClient
from .tools import TOOLS_SCHEMA, execute
from .wechat_profile import load_profile


def _clean_reply(text: str) -> str:
    """折叠复读、限制 @ 数量，避免微调模型输出“@某人@某人…”刷屏。"""
    text = (text or "").strip()
    text = re.sub(r"(.{1,12}?)\1{3,}", r"\1", text)  # 连续重复片段只留一次
    if text.count("@") > 2:  # 最多保留 2 个 @
        idx = -1
        for _ in range(2):
            idx = text.find("@", idx + 1)
        text = text[:idx]
    return text.strip()


def _is_degenerate(text: str) -> bool:
    """判断是否是复读/刷屏式输出。"""
    if len(text) > 120 or text.count("@") > 4:
        return True
    return bool(re.search(r"(.{2,12}?)\1{3,}", text))


def _similar(a: str, b: str) -> bool:
    """两句话是否高度相似（用于防复读）。"""
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


class CloneAgent:
    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.root = root
        self.llm = OllamaClient(
            base_url=cfg["ollama"]["base_url"],
            chat_model=cfg["ollama"]["chat_model"],
            embed_model=cfg["ollama"]["embed_model"],
        )
        self.persona = self._load_persona()
        # web/RAG 问答是否开启 thinking（微信回复由 chat_native 单独控制）
        self.llm.think_default = bool(cfg.get("thinking_answer", False))
        self.llm.num_ctx = int(cfg.get("num_ctx", 8192))
        self.kb = KnowledgeBase(root / cfg["vector_store_file"])
        self.memory = LongTermMemory(root / cfg["memory_file"])
        self.top_k = int(cfg.get("top_k", 4))
        self.max_tool_rounds = int(cfg.get("max_tool_rounds", 6))
        self.temperature = float(cfg.get("temperature", 0.7))

    # ------------------------------------------------------------------ #
    def _load_persona(self) -> str:
        p = self.root / self.cfg["persona_file"]
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        return "你是一个乐于助人的 AI 分身。"

    def _build_system(self) -> str:
        now = time.strftime("%Y-%m-%d %H:%M:%S %A")
        recent = self.memory.recent(20)
        mem_block = ""
        if recent:
            lines = [f"- [{r['role']}] {str(r['content'])[:300]}" for r in recent]
            mem_block = "【长期记忆 · 最近沉淀】\n" + "\n".join(lines) + "\n"
        return (
            f"{self.persona}\n\n"
            f"当前时间：{now}\n\n"
            f"{mem_block}"
            "你有工具可用：需要当前时间、读写项目目录内文件、或记住长期信息时调用工具；"
            "其余情况直接回答，不要为了用工具而用工具。"
        )

    # ------------------------------------------------------------------ #
    def _retrieve(self, query: str) -> list[dict]:
        if not self.kb.items:
            return []
        try:
            emb = self.llm.embed([query])[0]
        except Exception:
            return []  # 嵌入失败不阻断正常聊天
        return self.kb.search(emb, self.top_k)

    # ------------------------------------------------------------------ #
    def answer(self, user_text: str) -> str:
        hits = self._retrieve(user_text)
        context_block = ""
        if hits:
            parts = [h["text"] for h in hits]
            context_block = (
                "【以下是关于我的背景信息。请把它们当作你本来就知道的事，"
                "自然地用第一人称融入回答；不要提及“资料”“知识库”“来源”等字眼，"
                "也不要说自己是通过资料了解到的】\n"
                + "\n---\n".join(parts)
                + "\n\n"
            )

        user_content = (context_block + user_text) if context_block else user_text
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system()},
            {"role": "user", "content": user_content},
        ]

        final_text = ""
        for _ in range(self.max_tool_rounds):
            msg = self.llm.chat(messages, tools=TOOLS_SCHEMA, temperature=self.temperature)
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    try:
                        result = execute(tc, self.memory)
                    except Exception as e:  # 工具异常不中断整个对话
                        result = f"工具执行出错：{e}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": result,
                        }
                    )
                continue

            # Qwen3.5 等「思考型」模型可能把内容放在 thinking 字段，做个兜底
            final_text = (
                msg.get("content")
                or msg.get("thinking")
                or msg.get("reasoning_content")
                or ""
            )
            break

        # 沉淀本轮对话到长期记忆
        self.memory.append("user", user_text)
        self.memory.append("assistant", final_text)
        return final_text

    # ------------------------------------------------------------------ #
    def _style_examples(self, chat: str, limit: int = 5) -> list[str]:
        """从该会话历史里取“我”说过的话，作为风格示例（不给模型背内容）。"""
        if not chat:
            return []
        try:
            from .wechat_store import load_messages, safe_name, strip_parens

            msgs = load_messages(safe_name(strip_parens(chat)))
        except Exception:
            return []
        bad = ("资料里", "资料库", "我不记得", "没有相关记录")
        out: list[str] = []
        seen: set[str] = set()
        for m in msgs:
            if m.get("role") != "me":
                continue
            t = (m.get("text") or "").strip()
            if len(t) < 3 or any(b in t for b in bad):
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out[-limit:]

    # ------------------------------------------------------------------ #
    def _analyze_intent(self, convo: str) -> dict:
        """先用模型判断对方意图（分享/倾诉/提问/邀请/指责/辱骂/寒暄）。"""
        prompt = (
            "你是对话意图分析器。根据下面的聊天记录，判断最后一条对方消息的意图。\n"
            "只输出 JSON："
            '{"intent":"分享/倾诉/提问/邀请/指责/辱骂/寒暄/调侃之一",'
            '"addressed_to_me":true或false（是否针对我，包括提到我、@我、说“这个AI/学长”），'
            '"expectation":"对方期待什么回应，一句话",'
            '"topic":"这一句在聊什么，尽量具体，比如“线上麻将局”",'
            '"entities":["对方提到的地名/产品/人物/视频，最多2个"]}\n\n'
            f"聊天记录：\n{convo}"
        )
        try:
            text = self.llm.chat_native(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
            ).strip()
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    def chat_reply(
        self, history: list[dict], max_sentences: int = 3, chat: str = ""
    ) -> str:
        """微信替聊模式：根据最近对话生成简短、口语化的回复。

        与 answer() 的区别：不注入个人知识库、不调用工具，只依据对话上下文，
        避免把“辞职信/工作经历”之类文档内容硬塞进日常聊天。
        chat 传入会话名时：该会话在 data/wechat 下没有历史语料，就退回基础模型。
        """
        recent = [m for m in history[-40:] if isinstance(m, dict) and m.get("text")]
        if not recent:
            return ""
        lines = []
        for m in recent:
            who = "我" if m.get("role") == "me" else "对方"
            lines.append(f"{who}：{str(m.get('text', '')).strip()}")
        convo = "\n".join(lines)
        style_examples = self._style_examples(chat)
        profile = load_profile(self.root, chat) if chat else ""

        system = (
            f"{self.persona}\n\n"
            "现在你要用我的微信，替我回复最近这条消息。硬性要求：\n"
            "- 回复要短：总共最多 2 句，尽量控制在 40 个汉字以内；\n"
            "- 结构：先回应对方的话；只有自然需要时才补一句反问，不要每句都反问；\n"
            "- 口语化、自然，像真人发微信；\n"
            "- 不要长篇大论、不要分点、不要解释、不要复述对方的话；\n"
            "- 不要提资料、知识库、来源，不要暴露你是 AI；\n"
            "- 具体信息（时间、地点、安排、别人的事）如果历史/画像里没有就不要编造，"
            "用“再看吧”“再说”“我问下”这类话带过；历史/画像里有就照实说；\n"
            "- 对方问起某个人/某件事时，结合【画像】和上下文说具体一点，"
            "不要用“确实挺有意思”“这地方不错”这类总结/评价句；"
            "要短、留白、带点梗（如“有点说法”“有点东西”“懂的都懂”）；\n"
            "- 对方提到地名/产品/人物/视频时，如果你有把握，接一句不超过 10 个字的"
            "梗、冷知识或类比（例如把江汉路说成“小将军路”）；没把握就别提，不要编；\n"
            "- 不要主动约时间、提议聚聚，除非对方明确邀请；\n"
            "- 只输出要发送的回复内容本身，不要加引号。"
        )
        if style_examples:
            system += (
                "\n\n下面是我以前说过的话，请模仿我的语气、句长和用词"
                "（只学怎么说，不要照抄内容）：\n"
                + "\n".join(f"- {t}" for t in style_examples)
            )
        if profile:
            system += f"\n\n【对话对象/群的画像】\n{profile}"
        snippets: list = []
        if chat:
            try:
                from .wechat_history_rag import retrieve

                query = " ".join(
                    str(m.get("text", "")) for m in recent[-3:]
                ).strip()
                snippets = retrieve(
                    self.root,
                    self.llm,
                    chat,
                    query,
                    topk=int(self.cfg.get("history_topk", 5)),
                    min_score=float(self.cfg.get("history_min_score", 0.55)),
                    window=int(self.cfg.get("history_window", 1)),
                    days=int(self.cfg.get("history_days", 3)),
                    max_messages=int(self.cfg.get("history_max_messages", 999)),
                )
            except Exception:
                snippets = []
            if snippets:
                try:
                    rules = json.loads(
                        (self.root / "wechat_rules.json").read_text(encoding="utf-8")
                    )
                    me_names = set(rules.get("my_names") or [])
                except Exception:
                    me_names = set()
                system += (
                    "\n\n【相关历史（供参考，不要照抄）】\n"
                    + "\n".join(
                        f"- [{b.get('ts', '')}]\n"
                        + "\n".join(
                            f"  {'我' if ln['sender'] in me_names else ln['sender']}"
                            f"：{ln['text']}"
                            for ln in b["lines"]
                        )
                        for b in snippets
                    )
                    + "\n这些历史里能回答对方的内容，就据此回答，"
                    "不要说自己忘了；只有历史里确实没有才说不清楚。"
                    "不要原样复述上面的句子，用你自己的话概括。"
                )
        intent = self._analyze_intent(convo)
        # 邀约类：只有群聊、且明确出现我的名字/@ 才算邀请我本人
        if intent and intent.get("intent") == "邀请":
            try:
                rules = json.loads(
                    (self.root / "wechat_rules.json").read_text(encoding="utf-8")
                )
                me_names = set(rules.get("my_names") or [])
                groups = set(rules.get("groups") or [])
            except Exception:
                me_names = set()
                groups = set()
            is_group = chat in groups or "群" in chat or chat == "🐦🐦"
            last_text = str(recent[-1].get("text", "")) if recent else ""
            explicit = "@" in last_text or any(
                n and n in last_text for n in me_names
            )
            if is_group and not explicit:
                return ""
        if intent:
            system += (
                "\n\n【对方意图（模型分析）】\n"
                f"- 意图：{intent.get('intent', '')}\n"
                f"- 是否在对我说：{intent.get('addressed_to_me', '')}\n"
                f"- 期待：{intent.get('expectation', '')}\n"
                f"- 话题：{intent.get('topic', '')}\n"
                "请据此自然回应：分享/感慨就认同或补充（如“确实不错”“下次去看看”），"
                "不要追问时间/安排；提问就回答；邀请才确认时间；倾诉就安慰；"
                "如果对方在邀约、而画像里说我对这事不感兴趣，就委婉拒绝"
                "（如“不来，你们玩”），不要硬接；"
                "调侃/玩笑就自嘲或反击（比如被说“2块把你踢了”，可回“2块太少，起码5块”），"
                "不要顺着同意、不要转移话题、不要辩解，"
                "也不要提历史里的地点或无关话题；"
                "指责/辱骂就冷静简短。不要机械反问。"
            )
        # 名词速查：对方提到地名/产品/人物时给一句背景；调侃/玩笑不做速查
        entities = (
            intent.get("entities")
            if intent and intent.get("intent") != "调侃"
            else None
        )
        if entities:
            try:
                from .web_lookup import lookup

                ent = str(entities[0]).strip()
                note = lookup(ent, self.root / "data" / "entity_cache.json")
                if note:
                    system += (
                        f"\n\n【名词速查】{ent}：{note}\n"
                        "可据此接一句不超过 10 个字的梗或类比，不要照搬、不要展开。"
                    )
            except Exception:
                pass
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"最近的对话：\n{convo}\n\n请直接写出我接下来要发的一条回复。"},
        ]
        model = self.cfg.get("chat_reply_model") or None
        chat_models = self.cfg.get("chat_models") or {}
        if chat and chat in chat_models:
            # 该会话指定了专属人格模型
            model = chat_models[chat] or None
        elif model and chat:
            from .wechat_store import safe_name, strip_parens

            name = safe_name(strip_parens(chat))
            if not (self.root / "data" / "wechat" / name).exists():
                model = None  # 没有该会话的历史语料 → 用基础模型回答
        reply = self.llm.chat_native(
            messages,
            model=model,
            temperature=0.7,
            think=bool(self.cfg.get("thinking", False)),
        ).strip()
        degenerate = _is_degenerate(reply)

        # 微调模型拒绝/空回复/复读时，回退到基础模型再试一次
        refusals = (
            "资料里",
            "资料中没有",
            "知识库里",
            "资料库",
            "我不记得",
            "没有相关记录",
        )
        if model and (
            len(reply) < 2 or degenerate or any(k in reply for k in refusals)
        ):
            fallback = self.llm.chat_native(messages, model=model, temperature=0.7)
            alt = fallback.strip()
            if len(alt) >= 2 and not _is_degenerate(alt):
                reply = alt
        reply = _clean_reply(reply)
        # 去掉书面化/直白的评价，换成留白说法
        for a, b in (
            ("这地儿不错", "有点说法"),
            ("这地方不错", "有点说法"),
            ("这地方确实", "这地方"),
            ("确实挺有意思", "有点说法"),
            ("挺有意思", "有点东西"),
        ):
            reply = reply.replace(a, b)
        # 去掉主动约时间/提议聚聚的句子（对方没明确邀请时）
        parts = [p for p in re.split(r"(?<=[。！？!?])", reply) if p.strip()]
        planning = ("啥时候", "几点", "时间咋样", "聚聚", "约", "安排")
        if len(parts) > 1 and any(k in parts[-1] for k in planning):
            parts = parts[:-1]
        reply = "".join(parts)

        # 防复读：与最近我说过的话高度相似时，换基础模型重答
        recent_mine = [
            str(m.get("text", "")).strip()
            for m in recent
            if m.get("role") == "me" and str(m.get("text", "")).strip()
        ]
        if reply and any(_similar(reply, t) for t in recent_mine[-3:]):
            try:
                alt = self.llm.chat_native(messages, model=model, temperature=0.8)
                alt_text = _clean_reply(alt.strip())
                if alt_text and not any(
                    _similar(alt_text, t) for t in recent_mine[-3:]
                ):
                    reply = alt_text
            except Exception:
                pass

        # 防止复述历史原句
        if snippets:
            hist_lines = [
                ln["text"]
                for b in snippets
                for ln in b["lines"]
                if len(ln.get("text", "")) >= 4
            ]
            hist_parts = [
                p.strip()
                for t in hist_lines
                for p in re.split(r"[。！？!?，,]", t)
                if len(p.strip()) >= 5
            ]
            quoted = any(_similar(reply, t) for t in hist_lines) or any(
                p in reply for p in hist_parts
            )
            if quoted:
                try:
                    retry = [
                        messages[0],
                        {
                            "role": "user",
                            "content": messages[1]["content"]
                            + "\n\n要求：不要复述历史原句，用你自己的话概括或回答。",
                        },
                    ]
                    alt = self.llm.chat_native(retry, model=model, temperature=0.8)
                    alt_text = _clean_reply(alt.strip())
                    if alt_text and not any(
                        _similar(alt_text, t) for t in hist_lines
                    ):
                        reply = alt_text
                except Exception:
                    pass

        # 兜底：若模型还是写长了，按句号/问号切成最多 2 句
        parts = re.split(r"(?<=[。！？!?])", reply)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 2:
            reply = "".join(parts[:2])
        return reply
