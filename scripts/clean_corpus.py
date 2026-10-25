"""清洗微信语料：去掉系统消息、纯 @、低信息短句、重复，保留有对话价值的内容。

用法：
    python scripts/clean_corpus.py            # 只统计（dry-run）
    python scripts/clean_corpus.py --apply    # 真正写回
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.wechat_store import STORE, export_markdown, load_messages, safe_name  # noqa: E402

UI_PATTERNS = (
    "撤回了一条消息", "加入了群聊", "邀请", "拍了拍", "以上是打招呼",
    "你已添加", "消息已发出", "对方正在输入", "该消息类型暂不支持",
    "红包", "转账", "拍一拍", "语音", "视频通话", "已取消",
    "资料里没有", "资料中没有", "知识库里没有", "我不记得",
)
LOW_INFO = {
    "嗯", "哦", "啊", "哈", "好", "行", "对", "是", "在", "？", "。",
    "...", "1", "6", "收到", "ok", "OK", "哈哈", "哈哈哈", "？？", "。。。",
}
URL_RE = re.compile(r"https?://\S+")

# 同名归一（OCR/昵称变体）
NAME_ALIASES = {
    "张驰": "张弛",
    "小江、": "小江",
    "候奕丰": "侯亦丰",
    "土宇航": "王宇航",
    "BSEAN": "闫啸SEAN",
    "Richer （人机版）": "Richer",
    "Richer（人机版）": "Richer",
    "Richer (人机版）": "Richer",
    "Richer(人机版)": "Richer",
    "吴人机": "Richer",
}


def worth_keeping(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if any(p in t for p in UI_PATTERNS):
        return False
    if t in LOW_INFO:
        return False
    # 纯 @ 名单
    if t.count("@") >= 2 and len(t) < 40:
        return False
    # 纯链接
    if URL_RE.fullmatch(t):
        return False
    # 没有任何文字/数字（纯表情/符号）
    if not re.search(r"[\w\u4e00-\u9fff]", t):
        return False
    # 单字符
    if len(t) <= 1:
        return False
    # 长度 ≤2 且只有数字/符号
    if len(t) <= 2 and re.fullmatch(r"[\d\W_]+", t):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正写回文件")
    args = ap.parse_args()

    total_before = total_after = 0
    for d in sorted(STORE.iterdir()):
        if not d.is_dir():
            continue
        chat = d.name
        msgs = load_messages(chat)
        if not msgs:
            continue

        kept: list[dict] = []
        recent: list[tuple] = []
        for m in msgs:
            text = (m.get("text") or "").strip()
            if not worth_keeping(text):
                continue
            sender = (m.get("sender") or "").strip()
            if not sender or sender in ("未知", "None"):
                continue
            m["sender"] = NAME_ALIASES.get(sender, sender)
            key = (m.get("sender"), text)
            if key in recent[-5:]:  # 近期重复
                continue
            recent.append(key)
            kept.append(m)

        before, after = len(msgs), len(kept)
        total_before += before
        total_after += after

        senders_before = Counter(m.get("sender") for m in msgs)
        senders_after = Counter(m.get("sender") for m in kept)
        print(f"\n{chat}: {before} -> {after}")
        for name, cnt in senders_after.most_common():
            print(f"  {name}: {senders_before.get(name, 0)} -> {cnt}")

        if args.apply and after != before:
            p = d / "messages.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for m in kept:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            export_markdown(chat)

            raw = ROOT / "data" / "raw" / "wechat" / f"{safe_name(chat)}.md"
            raw.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"# {chat} 聊天记录", ""]
            for m in kept:
                lines.append(f"- [{m.get('ts', '')}] {m.get('sender', '')}：{m.get('text', '')}")
            raw.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n总计: {total_before} -> {total_after}")
    if not args.apply:
        print("（dry-run，加 --apply 才会写回）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
