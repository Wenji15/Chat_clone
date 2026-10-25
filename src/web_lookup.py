"""轻量名词速查：Bing 搜索取首条摘要，2 秒超时，带缓存。

用于"对方提到地名/产品/人物"时给模型一个简短背景，便于接一句梗。
"""
from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

_CACHE: dict[str, str] = {}


def lookup(entity: str, cache_path: Path | None = None, timeout: float = 2.0) -> str:
    key = (entity or "").strip()
    if not key or len(key) > 24:
        return ""
    if key in _CACHE:
        return _CACHE[key]

    # 先查本地备注（data/entity_notes.json），适合本地小地名/熟人梗
    notes_path = cache_path.parent / "entity_notes.json" if cache_path else None
    if notes_path and notes_path.exists():
        try:
            notes = json.loads(notes_path.read_text(encoding="utf-8"))
            for k, v in notes.items():
                if k and k in key:
                    _CACHE[key] = str(v)
                    return str(v)
        except Exception:
            pass

    if cache_path and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if key in data:
                _CACHE[key] = data[key]
                return data[key]
        except Exception:
            pass

    url = "https://cn.bing.com/search?q=" + urllib.parse.quote(key)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", "ignore")
    except Exception:
        return ""

    m = re.search(r'<li class="b_algo".*?<h2>(.*?)</h2>(.*?)</li>', text, re.S)
    if not m:
        return ""
    title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    body = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
    note = (title + "：" + body)[:120]
    _CACHE[key] = note
    if cache_path:
        try:
            data = (
                json.loads(cache_path.read_text(encoding="utf-8"))
                if cache_path.exists()
                else {}
            )
            data[key] = note
            cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except Exception:
            pass
    return note
