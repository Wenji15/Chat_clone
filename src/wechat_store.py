"""微信聊天记录本地存储层。

规则 2：单聊每人一个目录、群聊每群一个目录；记录发送人、内容、时间。
目录结构：
    data/wechat/<会话名>/
        messages.jsonl   # 结构化记录，一行一条
        chat.txt          # 可读导出（纯文本）
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "wechat"


def safe_name(name: str) -> str:
    """把会话名转成安全的目录名。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name or "").strip("_")
    return cleaned or "unknown"


def strip_parens(name: str) -> str:
    """去掉名字里的括号及括号内容，例如 小鸽哥踢球群（112） -> 小鸽哥踢球群。"""
    return re.sub(r"[（(][^)）]*[)）]", "", name or "").strip()


_SYSTEM_NAMES = {
    "拍即传", "微信团队", "服务号", "订阅号", "文件传输助手", "小程序",
    "视频号", "微信支付", "腾讯新闻", "群助手", "微信运动",
}
_NAMES_CACHE: tuple[set[str], dict[str, str]] | None = None


def _norm_name(name: str) -> str:
    s = re.sub(r"\s+", "", name or "")
    s = s.replace("(", "（").replace(")", "）")
    return s.strip("：:。，,、")


def _load_names() -> tuple[set[str], dict[str, str]]:
    """读取 data/people/names.txt：每行“标准名”或“标准名|别名1,别名2”。"""
    global _NAMES_CACHE
    if _NAMES_CACHE is not None:
        return _NAMES_CACHE
    names: set[str] = set()
    alias: dict[str, str] = {}
    p = ROOT / "data" / "people" / "names.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                std, rest = line.split("|", 1)
                std = _norm_name(std)
                names.add(std)
                for a in re.split(r"[|,，]", rest):
                    a = _norm_name(a)
                    if a:
                        alias[a] = std
                        alias[a.lower()] = std
            else:
                names.add(_norm_name(line))
    _NAMES_CACHE = (names, alias)
    return _NAMES_CACHE


def clean_sender(name: str) -> str:
    """清洗发送人名：去编号/标点、归一别名、拒绝长句/系统名/数字名。"""
    s = _norm_name(name)
    s = re.sub(r"^\d{1,3}[.、]", "", s)
    s = s.strip("：:。，,、")
    if not s:
        return ""
    names, alias = _load_names()
    if s in alias:
        return alias[s]
    if s.lower() in alias:
        return alias[s.lower()]
    if s in names:
        return s
    lower_names = {n.lower(): n for n in names}
    if s.lower() in lower_names:
        return lower_names[s.lower()]
    if s in _SYSTEM_NAMES:
        return ""
    if any(k in s for k in ("公众号", "小程序", "视频号", "文件传输", "微信团队", "客服")):
        return ""
    if len(s) > 8:
        return ""
    import difflib

    hit = difflib.get_close_matches(s, names, n=1, cutoff=0.85) if names else []
    if hit:
        return hit[0]
    # 明显是句子/系统提示的，不当人名
    if re.search(r"[的了呢吧啊呀哦嘛]$", s):
        return ""
    if any(
        k in s
        for k in ("需要", "确认", "为准", "详情", "生效", "调查", "赛季", "会员", "费用")
    ):
        return ""
    if re.fullmatch(r"[\u4e00-\u9fa5A-Za-z·（）]{1,8}", s):
        return s
    return ""


def chat_dir(chat: str) -> Path:
    d = STORE / safe_name(chat)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_messages(chat: str) -> list[dict]:
    p = chat_dir(chat) / "messages.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _data_heavy(text: str) -> bool:
    """长文本里数字很多、或接龙/报名名单 → 统一记为“报名”。"""
    if len(text) < 10:
        return False
    digits = sum(ch.isdigit() for ch in text)
    if digits >= 10 and digits / len(text) >= 0.2:
        return True
    if ("接龙" in text or "报名" in text) and len(
        re.findall(r"\d{1,2}[.、]", text)
    ) >= 2:
        return True
    if len(re.findall(r"\d{1,2}[.、]", text)) >= 3:
        return True
    return False


def append_messages(chat: str, msgs: list[dict]) -> int:
    """追加消息，按 (ts, sender, text) 去重（无 ts 时退化为 (sender, text)）。

    实时采集时同一屏会被多次识别，必须靠时间+发送人+内容三元组去重，
    否则重复的“行/好/哈哈”会被误删或重复写入。
    """
    existing = load_messages(chat)
    # 老数据没有 seq：按当前顺序补一个，保证后续同分钟消息能稳定排序
    for i, m in enumerate(existing, 1):
        m.setdefault("seq", i)
    max_seq = max((int(m.get("seq") or 0) for m in existing), default=0)
    seen = {
        (str(m.get("ts") or ""), m.get("sender") or "", m.get("text") or "")
        for m in existing
    }
    all_msgs = list(existing)
    added = 0
    for m in msgs:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if _data_heavy(text):
            text = "报名"
        ts = str(m.get("ts") or time.strftime("%Y-%m-%d %H:%M"))
        key = (ts, m.get("sender") or "", text)
        if key in seen:
            continue
        max_seq += 1
        rec = {
            "seq": max_seq,
            "ts": ts,
            "role": m.get("role"),          # me / them
            "sender": clean_sender(m.get("sender")) or None,
            "text": text,
        }
        if m.get("source"):
            rec["source"] = m["source"]
        all_msgs.append(rec)
        seen.add(key)
        added += 1

    # 同一分钟内按 seq 保序（分钟级时间戳无法区分先后）
    all_msgs.sort(key=lambda m: (str(m.get("ts", "")), int(m.get("seq") or 0)))
    p = chat_dir(chat) / "messages.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for m in all_msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return added


def export_text(chat: str) -> Path:
    """导出为纯文本 chat.txt（对齐聊天记录窗口：新消息在上，旧消息在下）。

    注意：messages.jsonl 内部仍按时间正序（旧→新），只有导出文件是倒序。
    """
    msgs = sorted(
        load_messages(chat),
        key=lambda m: (str(m.get("ts", "")), int(m.get("seq") or 0)),
        reverse=True,
    )
    out = chat_dir(chat) / "chat.txt"
    lines = []
    for m in msgs:
        who = m.get("sender") or ("我" if m.get("role") == "me" else "对方")
        lines.append(f"[{m.get('ts', '')}] {who}：{m.get('text', '')}")
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


# 兼容旧调用名
export_markdown = export_text
