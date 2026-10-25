"""RAG 查询：在个人知识库里检索相关内容，再让本地模型基于资料回答。

用法：
    python scripts/query.py "闫总在墨水湖说了什么"
    python scripts/query.py "..." --topk 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.memory import KnowledgeBase  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config()
    kb = KnowledgeBase(ROOT / cfg["vector_store_file"])
    if not kb.items:
        print("知识库为空，请先运行：python scripts/ingest.py")
        return 1

    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        chat_model=cfg["ollama"]["chat_model"],
        embed_model=cfg["ollama"]["embed_model"],
    )
    emb = client.embed([args.question])[0]
    hits = kb.search(emb, args.topk)

    context = "\n---\n".join(h["text"] for h in hits)
    messages = [
        {
            "role": "system",
            "content": (
                "你是我的个人分身。请只根据提供的资料回答；"
                "资料不足时，用常识简洁回答，不要编造具体事实；"
                "不要输出“资料里没有”这类话。"
            ),
        },
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{args.question}"},
    ]
    msg = client.chat(messages, temperature=0.3)
    print(msg.get("content") or "（无回复）")

    print("\n--- 命中片段 ---")
    for h in hits:
        src = h.get("meta", {}).get("source", "")
        print(f"[{h.get('score')}] {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
