"""清洗历史语料里的发送人名（老数据迁移）。

用法：
    python scripts/normalize_senders.py          # 预览
    python scripts/normalize_senders.py --apply  # 写回（每会话先备份 .bak）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wechat_store import (  # noqa: E402
    STORE,
    clean_sender,
    export_text,
    load_messages,
    safe_name,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写回（默认只预览）")
    args = ap.parse_args()

    total = changed = dropped = 0
    for d in sorted(STORE.iterdir()):
        if not d.is_dir():
            continue
        chat = d.name
        msgs = load_messages(chat)
        if not msgs:
            continue
        c = dp = 0
        for m in msgs:
            old = (m.get("sender") or "").strip()
            new = clean_sender(old)
            if new != old:
                c += 1
                if old and not new:
                    dp += 1
            m["sender"] = new or None
        total += len(msgs)
        changed += c
        dropped += dp
        print(f"{chat}: {len(msgs)} 条，变更 {c}，丢弃 {dp}")

        if args.apply and c:
            p = d / "messages.jsonl"
            bak = d / f"messages.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            p.replace(bak)
            with open(p, "w", encoding="utf-8") as f:
                for m in msgs:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            export_text(chat)
            # 同步重建 raw md（新→旧）
            raw = ROOT / "data" / "raw" / "wechat" / f"{safe_name(chat)}.md"
            raw.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"# {chat} 聊天记录", ""]
            for m in reversed(msgs):
                who = m.get("sender") or ("我" if m.get("role") == "me" else "对方")
                lines.append(f"- [{m.get('ts', '')}] {who}：{m.get('text', '')}")
            raw.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n合计 {total} 条，变更 {changed}，丢弃 {dropped}")
    if not args.apply:
        print("预览模式；加 --apply 写回")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
