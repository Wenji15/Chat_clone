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
    """追加消息，按 (ts, sender, text) 去重（无 ts 时退化为 (sender, text)）。

    实时采集时同一屏会被多次识别，必须靠时间+发送人+内容三元组去重，
    否则重复的“行/好/哈哈”会被误删或重复写入。
    """
    existing = load_messages(chat)
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
        ts = str(m.get("ts") or time.strftime("%Y-%m-%d %H:%M"))
        key = (ts, m.get("sender") or "", text)
        if key in seen:
            continue
        rec = {
            "ts": ts,
            "role": m.get("role"),          # me / them
            "sender": m.get("sender"),      # 具体发送人（群聊用）
            "text": text,
        }
        if m.get("source"):
            rec["source"] = m["source"]
        all_msgs.append(rec)
        seen.add(key)
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
