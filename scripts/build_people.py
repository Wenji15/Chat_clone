"""按标准名生成个人档案：data/people/<名>.md

用法：
    python scripts/build_people.py                 # 所有名单内的人
    python scripts/build_people.py --person 田政
    python scripts/build_people.py --min-messages 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402
from src.wechat_store import (  # noqa: E402
    STORE,
    _load_names,
    clean_sender,
    is_me_name,
    load_messages,
    safe_name,
)

CHUNK = 40


def collect_by_person(
    chats: set[str] | None = None, context: int = 2
) -> tuple[dict[str, list[dict]], dict[str, dict], int]:
    names, alias = _load_names()
    out: dict[str, list[dict]] = {n: [] for n in names}
    stats: dict[str, dict] = {}
    total = 0
    for d in sorted(STORE.iterdir()):
        if not d.is_dir():
            continue
        chat = d.name
        if chats and chat not in chats:
            continue
        msgs = load_messages(chat)
        total += len(msgs)
        for i, m in enumerate(msgs):
            if m.get("role") != "them":
                continue
            name = clean_sender(m.get("sender"))
            if not name or is_me_name(m.get("sender")):
                continue
            win = msgs[max(0, i - context) : i + context + 1]
            lines = []
            for x in win:
                who = (
                    "我"
                    if x.get("role") == "me" or is_me_name(x.get("sender"))
                    else (clean_sender(x.get("sender")) or "成员")
                )
                lines.append(f"{who}：{x.get('text', '')}")
            out.setdefault(name, []).append(
                {
                    "chat": chat,
                    "ts": m.get("ts", ""),
                    "ctx": "\n".join(lines),
                }
            )
            st = stats.setdefault(
                name, {"count": 0, "days": set(), "first": "", "last": ""}
            )
            st["count"] += 1
            ts = str(m.get("ts") or "")
            if ts:
                st["days"].add(ts[:10])
                if not st["first"] or ts < st["first"]:
                    st["first"] = ts
                if ts > st["last"]:
                    st["last"] = ts
    return out, stats, total


def summarize_chunk(
    client: OllamaClient, name: str, rows: list[dict], st: dict
) -> str:
    body = "\n---\n".join(r["ctx"] for r in rows)
    days = len(st.get("days") or [])
    meta = (
        f"该人共有 {st.get('count', 0)} 条消息，"
        f"时间范围 {st.get('first', '')} ~ {st.get('last', '')}，活跃天数 {days}。"
        "片段只是抽样，不要据此判断长期活跃度；只有统计上明显高频才可称“活跃”。"
    )
    prompt = (
        "下面是一个微信群里的聊天片段，每条前面是说话人，"
        f"「我」是吴恩吉，「{name}」是我们要分析的人。\n"
        f"{meta}\n"
        "重要规则：\n"
        f"1. {name} 说的话不一定是对「我」说的，可能是在回别人或自言自语；\n"
        "2. 只有明确 @我、回复我、或与我连续对话的内容，才算「对我的态度」；\n"
        "3. 不要推断恋爱、暧昧、性别；没有明确依据就写“不确定”；\n"
        "4. 区分“对群友的态度”和“对我的态度”；\n"
        "5. 消息少/活跃天数少时，写“样本少，只代表该时段”，不要称核心成员。\n"
        "请提炼：\n"
        "- 关系/身份\n- 性格\n- 说话风格（句长、口头禅、梗）\n"
        "- 常聊话题\n- 对群友的态度\n- 关系网（和群内其他人的关系）\n"
        "- 对我的态度（仅直接互动）\n"
        "- 活跃度（仅基于统计，不夸大）\n"
        "- 近期状态\n- 有代表性的原话 1-3 条\n"
        "不确定的写“不确定”，不要编造。用简洁中文条目。\n\n"
        + body
    )
    return client.chat_native(
        [{"role": "user", "content": prompt}], temperature=0.3
    ).strip()


def finalize(
    client: OllamaClient, name: str, notes: str, st: dict, span: str
) -> str:
    days = len(st.get("days") or [])
    count = int(st.get("count", 0) or 0)
    share = st.get("share", 0)
    meta = f"消息数：{count}，时间范围：{span}，活跃天数：{days}，占已分析消息：{share}%"
    prompt = (
        f"请把下面关于「{name}」的零散笔记合并成一份档案（Markdown）。"
        "保留有依据的内容，去掉重复；不确定的标注“不确定”。"
        "不要推断恋爱/暧昧/性别；没有直接互动证据时，“对我的态度”写“不确定”。\n"
        f"样本统计：{meta}\n"
        "如果消息数 < 100 或活跃天数 < 5，必须在“活跃度”里写"
        "“样本少，仅代表该时段有发言”，禁止称“核心成员/活跃分子/长期活跃”。\n"
        "低样本时，全文禁止使用“核心成员、重要成员、活跃分子、核心球友、活跃球员”"
        "等表述，只能写“该时段有发言的群友”。\n"
        "结构：关系、性格、说话风格、常聊话题、对群友的态度、"
        "关系网（和群内其他人的关系）、对我的态度（仅直接互动）、"
        "活跃度（按统计写）、近期状态、口头禅/梗、原话依据。\n\n"
        f"{notes}"
    )
    body = client.chat_native(
        [{"role": "user", "content": prompt}], temperature=0.3
    ).strip()
    if count < 100 or days < 5:
        for a, b in (
            ("核心成员", "群友"),
            ("重要成员", "群友"),
            ("活跃分子", "群友"),
            ("核心球友", "球友"),
            ("活跃球员", "群友"),
            ("长期活跃", "该时段有发言"),
        ):
            body = body.replace(a, b)
    return f"# {name}\n\n> {meta}\n\n{body}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="")
    ap.add_argument("--min-messages", type=int, default=30)
    ap.add_argument("--max-messages", type=int, default=200)
    ap.add_argument("--context", type=int, default=2, help="前后各带几条消息")
    ap.add_argument(
        "--chats",
        default="",
        help="只分析这些会话，逗号分隔；为空=全部",
    )
    args = ap.parse_args()

    cfg = load_config()
    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        chat_model=cfg["ollama"]["chat_model"],
        embed_model=cfg["ollama"]["embed_model"],
    )
    chats = {c.strip() for c in args.chats.split(",") if c.strip()} or None
    by_person, stats, total = collect_by_person(
        chats=chats, context=args.context
    )
    targets = [args.person] if args.person else sorted(by_person)

    outdir = ROOT / "data" / "people"
    outdir.mkdir(parents=True, exist_ok=True)

    for name in targets:
        rows = by_person.get(name) or []
        if len(rows) < args.min_messages:
            print(f"[skip] {name}: {len(rows)} 条 < {args.min_messages}")
            continue
        rows.sort(key=lambda r: str(r.get("ts", "")))
        rows = rows[-args.max_messages :]
        span = f"{rows[0]['ts']} ~ {rows[-1]['ts']}" if rows else ""
        st = dict(stats.get(name) or {})
        st["share"] = round((st.get("count") or 0) / max(1, total) * 100, 2)
        notes = []
        for i in range(0, len(rows), CHUNK):
            part = rows[i : i + CHUNK]
            notes.append(summarize_chunk(client, name, part, st))
            print(f"[{name}] {i + len(part)}/{len(rows)}")
        md = finalize(client, name, "\n\n".join(notes), st, span)
        path = outdir / f"{safe_name(name)}.md"
        if path.exists():
            path.replace(path.with_suffix(".md.bak"))
        path.write_text(md, encoding="utf-8")
        print(f"[ok] {name} -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
