"""微信聊天记录本地存储层。

规则 2：单聊每人一个目录、群聊每群一个目录；记录发送人、内容、时间。
目录结构：
    data/wechat/<会话名>/
        messages.jsonl   # 结构化记录，一行一条
        chat.md          # 可读导出（也可直接喂给 ingest.py）
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


def append_messages(chat: str, msgs: list[dict]) -> int:
    """追加消息，按 (role, text) 与最近 50 条做软去重，写回时按时间排序。

    排序保证“旧消息在上、新消息在下”。
    """
    existing = load_messages(chat)
    recent = {(m.get("role"), m.get("text")) for m in existing[-50:]}
    all_msgs = list(existing)
    added = 0
    for m in msgs:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        key = (m.get("role"), text)
        if key in recent:
            continue
        rec = {
            "ts": m.get("ts") or time.strftime("%Y-%m-%d %H:%M"),
            "role": m.get("role"),          # me / them
            "sender": m.get("sender"),      # 具体发送人（群聊用）
            "text": text,
        }
        all_msgs.append(rec)
        recent.add(key)
        added += 1

    all_msgs.sort(key=lambda m: str(m.get("ts", "")))
    p = chat_dir(chat) / "messages.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for m in all_msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return added


def export_markdown(chat: str) -> Path:
    """导出为可读 Markdown（按时间升序，新消息在下方）。"""
    msgs = sorted(load_messages(chat), key=lambda m: str(m.get("ts", "")))
    md = chat_dir(chat) / "chat.md"
    lines = [f"# {chat}", ""]
    for m in msgs:
        who = m.get("sender") or ("我" if m.get("role") == "me" else "对方")
        lines.append(f"- [{m.get('ts', '')}] {who}：{m.get('text', '')}")
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
