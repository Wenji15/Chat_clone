"""按会话的历史消息检索：bge-m3 向量 + 余弦相似度。

每个会话的消息向量缓存到 data/wechat/<chat>/vectors.json，
消息数变化时自动重建。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .wechat_store import load_messages, safe_name, strip_parens

_CACHE: dict[str, dict] = {}


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_index(root: Path, client, chat: str) -> dict:
    key = safe_name(strip_parens(chat))
    if key in _CACHE:
        return _CACHE[key]

    msgs = [m for m in load_messages(chat) if (m.get("text") or "").strip()]
    texts = [m["text"].strip() for m in msgs]
    p = root / "data" / "wechat" / key / "vectors.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cached = int(data.get("count", 0))
            if cached == len(texts):
                _CACHE[key] = data
                return data
            # 增量：只嵌入新增消息
            if 0 < cached < len(texts):
                new_texts = texts[cached:]
                new_vectors: list[list[float]] = []
                for i in range(0, len(new_texts), 32):
                    new_vectors.extend(client.embed(new_texts[i : i + 32]))
                data["texts"].extend(new_texts)
                data["vectors"].extend(new_vectors)
                data["meta"].extend(
                    [
                        {"sender": m.get("sender", ""), "ts": m.get("ts", "")}
                        for m in msgs[cached:]
                    ]
                )
                data["count"] = len(texts)
                p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                _CACHE[key] = data
                return data
        except Exception:
            pass

    vectors: list[list[float]] = []
    for i in range(0, len(texts), 32):
        vectors.extend(client.embed(texts[i : i + 32]))
    data = {
        "count": len(texts),
        "texts": texts,
        "vectors": vectors,
        "meta": [{"sender": m.get("sender", ""), "ts": m.get("ts", "")} for m in msgs],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _CACHE[key] = data
    return data


def retrieve(
    root: Path,
    client,
    chat: str,
    query: str,
    topk: int = 5,
    min_score: float = 0.55,
    window: int = 1,
) -> list[dict]:
    if not query.strip():
        return []
    try:
        data = _load_index(root, client, chat)
    except Exception:
        return []
    if not data.get("vectors"):
        return []
    q = client.embed([query])[0]
    scored = [(_cos(q, v), i) for i, v in enumerate(data["vectors"])]
    scored.sort(reverse=True)

    picked: list[tuple[float, int]] = []
    seen: set[str] = set()
    for score, i in scored:
        if score < min_score or len(picked) >= topk:
            break
        text = data["texts"][i]
        if text in seen:
            continue
        seen.add(text)
        picked.append((score, i))

    texts = data["texts"]
    meta = data["meta"]
    used: set[int] = set()
    blocks: list[dict] = []
    for score, i in picked:
        lo = max(0, i - window)
        hi = min(len(texts) - 1, i + window)
        idxs = [j for j in range(lo, hi + 1) if j not in used]
        if not idxs:
            continue
        used.update(idxs)
        blocks.append(
            {
                "score": round(score, 3),
                "ts": meta[i].get("ts", ""),
                "lines": [
                    {
                        "sender": meta[j].get("sender") or "未知",
                        "text": texts[j],
                    }
                    for j in idxs
                ],
            }
        )
    return blocks
