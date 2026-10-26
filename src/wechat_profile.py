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


def load_person(root: Path, name: str) -> str:
    """读取个人档案 data/people/<名>.md。"""
    if not name:
        return ""
    p = root / "data" / "people" / f"{safe_name(name)}.md"
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
