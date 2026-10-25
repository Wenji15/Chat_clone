"""读取会话的联系人画像（data/profiles/<chat>.md）。"""
from __future__ import annotations

from pathlib import Path

from .wechat_store import safe_name, strip_parens


def load_profile(root: Path, chat: str) -> str:
    if not chat:
        return ""
    p = root / "data" / "profiles" / f"{safe_name(strip_parens(chat))}.md"
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
