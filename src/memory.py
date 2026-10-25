"""记忆系统：个人知识库（RAG 检索）+ 长期记忆（对话沉淀）。

纯 Python 实现（不依赖 numpy 等任何第三方库），零依赖即可运行。
余弦相似度在「千级以下 chunk」的个人知识库规模下性能足够；
若日后数据量上万，再替换成 Chroma / Qdrant / FAISS。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cosine(a: list[float], b: list[float], na: float | None = None, nb: float | None = None) -> float:
    if na is None:
        na = _norm(a)
    if nb is None:
        nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


class KnowledgeBase:
    """个人知识库：保存「文本 + 向量」，支持按查询向量检索 top_k。"""

    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict[str, Any]] = []
        if path.exists():
            self.load()

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = data.get("items", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"items": self.items}, f, ensure_ascii=False, indent=2)

    def add(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        meta: dict[str, Any] | None = None,
    ) -> None:
        vec = [float(x) for x in embedding]
        self.items.append(
            {
                "id": chunk_id,
                "text": text,
                "embedding": vec,
                "norm": _norm(vec),
                "meta": meta or {},
                "created_at": time.time(),
            }
        )

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """返回按相似度降序的 top_k 条记录（附带 score 字段）。"""
        if not self.items:
            return []
        q = [float(x) for x in query_embedding]
        nq = _norm(q)
        scored: list[tuple[float, dict]] = []
        for it in self.items:
            nb = it.get("norm")
            if nb is None:  # 兼容旧格式（没有预存范数）
                nb = _norm(it["embedding"])
            scored.append((_cosine(q, it["embedding"], nq, nb), it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": round(score, 4), **item} for score, item in scored[:top_k]]


class LongTermMemory:
    """长期记忆：JSONL 追加式，记录对话沉淀与用户说过的关键信息。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, role: str, content: str) -> None:
        record = {"ts": time.time(), "role": role, "content": content}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
