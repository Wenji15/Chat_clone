"""把 data/raw/ 里的个人资料切片、嵌入并写入知识库。

支持的格式：.txt / .md / .markdown / .json / .csv / .docx
（.doc 旧版二进制格式不支持，请另存为 .docx 或 .txt）

用法（在项目根目录执行）：
    python scripts/ingest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.docx_reader import extract_text  # noqa: E402
from src.memory import KnowledgeBase  # noqa: E402
from src.ollama_client import OllamaClient, OllamaError  # noqa: E402

CHUNK_SIZE = 500  # 每个片段的字符数
CHUNK_OVERLAP = 80  # 相邻片段重叠的字符数
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv"}
DOCX_SUFFIXES = {".docx"}
SUPPORTED = TEXT_SUFFIXES | DOCX_SUFFIXES


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def read_document(fp: Path) -> str:
    """按扩展名读取文档正文。"""
    suffix = fp.suffix.lower()
    if suffix in DOCX_SUFFIXES:
        return extract_text(fp)
    return fp.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    cfg = load_config()
    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        chat_model=cfg["ollama"]["chat_model"],
        embed_model=cfg["ollama"]["embed_model"],
    )
    kb = KnowledgeBase(ROOT / cfg["vector_store_file"])
    # 每次导入前清空（内存中），成功保存时整体替换，避免重复运行造成重复片段；
    # 若中途嵌入失败直接 return，旧知识库文件不会被动到。
    kb.items = []
    raw_dir = ROOT / cfg["knowledge_dir"]

    if not raw_dir.exists():
        print(f"资料目录不存在：{raw_dir}")
        return 1

    files = [
        p
        for p in sorted(raw_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ]
    if not files:
        print("没有找到可导入的文件。请把个人资料（.txt / .md / .csv / .json / .docx）")
        print("放进 data/raw/ 目录后，重新运行本脚本。")
        return 1

    total_added = 0
    for fp in files:
        try:
            content = read_document(fp)
        except Exception as e:
            print(f"跳过 {fp.name}：{e}")
            continue
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"跳过 {fp.name}：没有可提取的文字")
            continue
        print(f"嵌入 {fp.name}：{len(chunks)} 个片段…")
        try:
            vectors = client.embed(chunks)
        except OllamaError as e:
            print(f"\n{e}\n")
            return 1
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            kb.add(
                chunk_id=f"{fp.name}#{i}",
                text=chunk,
                embedding=vec,
                meta={"source": str(fp.relative_to(ROOT))},
            )
            total_added += 1

    kb.save()
    print(f"\n✅ 完成，共入库 {total_added} 个片段 → {kb.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
