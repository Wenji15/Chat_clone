"""为每个会话生成“联系人画像”：性格、关系、态度、关键经历。

用法：
    python scripts/build_profiles.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402
from src.wechat_store import STORE, load_messages, safe_name  # noqa: E402

PROMPT = """根据下面的聊天记录，为「{chat}」写一份联系人画像。
要求：
1. 主要说话的人：各自的性格、说话方式（简短/直接/爱开玩笑等）
2. 他们和“我”的关系、熟悉程度
3. 他们对“我”的态度（尊重/调侃/亲近/客气等）
4. 关键经历或事实：工作、爱好、常聊话题、重要的事
5. 只写记录里能看出来的，不要编造；不确定就写“不确定”
6. 总字数不超过 300 字，用简洁的要点

聊天记录：
{text}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-messages", type=int, default=300)
    ap.add_argument("--chat", default="", help="只生成指定会话")
    args = ap.parse_args()

    cfg = load_config()
    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        chat_model=cfg["ollama"]["chat_model"],
        embed_model=cfg["ollama"]["embed_model"],
    )
    out_dir = ROOT / "data" / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in sorted(STORE.iterdir()):
        if not d.is_dir():
            continue
        chat = d.name
        if args.chat and chat != args.chat:
            continue
        msgs = load_messages(chat)
        if len(msgs) < 20:
            print(f"跳过 {chat}：消息太少（{len(msgs)}）")
            continue
        recent = msgs[-args.max_messages :]
        text = "\n".join(
            f"{m.get('sender', '?')}：{m.get('text', '')}" for m in recent
        )
        msg = client.chat(
            [{"role": "user", "content": PROMPT.format(chat=chat, text=text)}],
            temperature=0.3,
        )
        content = (msg.get("content") or "").strip()
        if not content:
            print(f"跳过 {chat}：模型无输出")
            continue
        p = out_dir / f"{safe_name(chat)}.md"
        p.write_text(content, encoding="utf-8")
        print(f"{chat} -> {p} ({len(content)} 字)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
