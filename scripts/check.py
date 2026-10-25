"""环境自检：确认 Ollama 可用、模型就绪、配置正确。

用法（在项目根目录执行）：
    python scripts/check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.ollama_client import OllamaClient, OllamaError  # noqa: E402


def main() -> int:
    cfg = load_config()
    print("== 配置 ==")
    print(f"chat_model   : {cfg['ollama']['chat_model']}")
    print(f"embed_model  : {cfg['ollama']['embed_model']}")
    print(f"base_url     : {cfg['ollama']['base_url']}")

    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        chat_model=cfg["ollama"]["chat_model"],
        embed_model=cfg["ollama"]["embed_model"],
    )

    print("\n== 连接与模型 ==")
    ok = True

    try:
        emb = client.embed(["hello world"])
        print(f"[OK] 嵌入模型可用（维度 {len(emb[0])}）")
    except OllamaError as e:
        ok = False
        print(f"[FAIL] 嵌入模型不可用：{e}")

    try:
        msg = client.chat([{"role": "user", "content": "只回复两个字：就绪"}])
        reply = (msg.get("content") or msg.get("thinking") or "").strip()
        print(f"[OK] 对话模型可用，回复：{reply[:80]!r}")
    except OllamaError as e:
        ok = False
        print(f"[FAIL] 对话模型不可用：{e}")

    kb_path = ROOT / cfg["vector_store_file"]
    if kb_path.exists():
        items = json.loads(kb_path.read_text(encoding="utf-8")).get("items", [])
        print(f"\n[OK] 知识库已存在，共 {len(items)} 个片段：{kb_path.name}")
        if not items:
            print("     提示：先运行 python scripts/ingest.py 导入资料")
    else:
        print("\n[INFO] 知识库尚未创建，请先运行 python scripts/ingest.py")

    print("\n结论：" + ("全部就绪 ✅" if ok else "存在失败项 ❌，请按上面提示处理"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
