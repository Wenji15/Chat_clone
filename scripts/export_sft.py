"""把 data/wechat 下的聊天记录导出为 SFT 微调格式（JSONL 对话对）。

两种模式：
- 默认：只学“我”的说话习惯（assistant = 我的回复）
- --all：为每个发送者各生成样本（学习群里每个人的说话习惯）

输出：
    data/finetune/sft.jsonl / sft_all.jsonl      （全量）
    data/finetune/sft_train.jsonl / sft_val.jsonl（按 --split 划分）

用法：
    python scripts/export_sft.py
    python scripts/export_sft.py --all
    python scripts/export_sft.py --all --split 0.9
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wechat_store import STORE, load_messages  # noqa: E402


def _parse_ts(s):
    try:
        return time.mktime(time.strptime(str(s)[:16], "%Y-%m-%d %H:%M"))
    except Exception:
        return 0.0


def merge_turns(msgs: list[dict]) -> list[dict]:
    """把连续同一发送者的消息合并成一个“轮次”。"""
    turns: list[dict] = []
    for m in msgs:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        sender = m.get("sender") or ""
        ts = m.get("ts")
        if turns and turns[-1]["sender"] == sender:
            turns[-1]["text"] += "\n" + text
            turns[-1]["end"] = ts
        else:
            turns.append({"sender": sender, "text": text, "start": ts, "end": ts})
    return turns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="为所有发送者各生成样本")
    ap.add_argument("--split", type=float, default=0.9, help="训练集比例")
    ap.add_argument(
        "--turns",
        action="store_true",
        help="按轮次导出：合并连续同人、不标注发言人、带时间间隔",
    )
    args = ap.parse_args()

    persona = ""
    p = ROOT / "persona.md"
    if p.exists():
        persona = p.read_text(encoding="utf-8").strip()

    samples: list[dict] = []
    if STORE.exists():
        for d in sorted(STORE.iterdir()):
            if not d.is_dir():
                continue
            chat = d.name
            msgs = load_messages(chat)

            if args.turns:
                turns = merge_turns(msgs)
                is_group = chat.endswith("群") or chat == "🐦🐦"
                scene = f"微信群「{chat}」" if is_group else f"私聊「{chat}」"
                for i, t in enumerate(turns):
                    if i == 0:
                        continue
                    target = (t.get("text") or "").strip()
                    if not target:
                        continue
                    if target.count("@") >= 3 and len(target) < 40:
                        continue
                    ctx = turns[max(0, i - 4) : i]
                    if not ctx:
                        continue
                    gap = 0.0
                    a, b = _parse_ts(ctx[-1].get("end")), _parse_ts(t.get("start"))
                    if a and b:
                        gap = b - a
                    user_lines = [c["text"] for c in ctx]
                    if gap > 1800:
                        user_lines.append(f"（距上条 {int(gap // 60)} 分钟）")
                    samples.append(
                        {
                            "messages": [
                                {"role": "system", "content": f"场景：{scene}。"},
                                {"role": "user", "content": "\n".join(user_lines)},
                                {"role": "assistant", "content": target},
                            ]
                        }
                    )
                continue

            for i, m in enumerate(msgs):
                text = (m.get("text") or "").strip()
                sender = (m.get("sender") or "").strip()
                if not text:
                    continue
                # 纯 @ 名单（如 "@a@b@c"）不学，避免模型学成 @ 刷屏
                if text.count("@") >= 3 and len(text) < 40:
                    continue
                if args.all:
                    if not sender or sender == "未知":
                        continue
                else:
                    if m.get("role") != "me":
                        continue
                ctx = msgs[max(0, i - 6) : i]
                if not ctx:
                    continue

                # 场景：群聊/私聊 + 对话对象
                others = [
                    x.get("sender")
                    for x in ctx
                    if x.get("sender") and x.get("sender") != sender
                ]
                other = others[-1] if others else ""
                is_group = chat.endswith("群") or chat == "🐦🐦"
                if is_group:
                    scene = f"微信群「{chat}」"
                    if other:
                        scene += f"，正在和「{other}」等人聊天"
                else:
                    scene = f"私聊「{chat}」"

                if args.all:
                    system = (
                        f"你正在扮演「{sender}」。场景：{scene}。"
                        "请按他/她平时的说话方式回复。"
                    )
                else:
                    system = f"{persona}\n\n场景：{scene}" if persona else f"场景：{scene}"
                user_text = "\n".join(
                    f"{x.get('sender', '?')}：{x.get('text', '')}" for x in ctx
                )
                samples.append(
                    {
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": text},
                        ]
                    }
                )

    out_dir = ROOT / "data" / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "sft_turns" if args.turns else ("sft_all" if args.all else "sft")

    full = out_dir / f"{base}.jsonl"
    with open(full, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    random.seed(42)
    random.shuffle(samples)
    n = max(1, int(len(samples) * args.split)) if samples else 0
    train, val = samples[:n], samples[n:]
    for name, data in [("train", train), ("val", val)]:
        path = out_dir / f"{base}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(data)} 条")

    print(f"总计 {len(samples)} 条 -> {full}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
