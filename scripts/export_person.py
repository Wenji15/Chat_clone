"""按人导出 SFT 数据：每个发送者一份，便于单独训练“像某个人说话”。

用法：
    python scripts/export_person.py --min-samples 200
    python scripts/export_person.py --min-samples 50
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wechat_store import STORE, load_messages, safe_name  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=200)
    ap.add_argument("--split", type=float, default=0.9)
    args = ap.parse_args()

    buckets: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()

    if STORE.exists():
        for d in sorted(STORE.iterdir()):
            if not d.is_dir():
                continue
            chat = d.name
            msgs = load_messages(chat)
            for i, m in enumerate(msgs):
                sender = (m.get("sender") or "").strip()
                text = (m.get("text") or "").strip()
                if not sender or not text:
                    continue
                ctx = msgs[max(0, i - 6) : i]
                if not ctx:
                    continue
                counts[sender] += 1
                others = [
                    x.get("sender")
                    for x in ctx
                    if x.get("sender") and x.get("sender") != sender
                ]
                other = others[-1] if others else ""
                is_group = chat.endswith("群") or chat == "🐦🐦"
                scene = f"微信群「{chat}」"
                if is_group and other:
                    scene += f"，正在和「{other}」等人聊天"
                elif not is_group:
                    scene = f"私聊「{chat}」"
                user_text = "\n".join(
                    f"{x.get('sender', '?')}：{x.get('text', '')}" for x in ctx
                )
                buckets[sender].append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    f"你正在扮演「{sender}」。场景：{scene}。"
                                    "请按他/她平时的说话方式回复。"
                                ),
                            },
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": text},
                        ]
                    }
                )

    out_dir = ROOT / "data" / "finetune" / "persons"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'发送者':<20} {'样本数':>6}  导出")
    for sender, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        if len(items) < args.min_samples:
            continue
        random.seed(42)
        random.shuffle(items)
        n = max(1, int(len(items) * args.split))
        train, val = items[:n], items[n:]
        base = safe_name(sender)
        for name, data in [("train", train), ("val", val)]:
            p = out_dir / f"{base}_{name}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for s in data:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{sender:<20} {len(items):>6}  -> {base}_train/val.jsonl")

    print(f"\n共 {len(buckets)} 个发送者，阈值 {args.min_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
