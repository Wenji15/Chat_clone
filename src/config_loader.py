"""配置加载与路径解析（零依赖，仅标准库）。"""
from __future__ import annotations

import json
from pathlib import Path

# 项目根目录 = 本文件的上一级目录
ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "chat_model": "qwen2.5:7b",
        "embed_model": "nomic-embed-text",
    },
    "persona_file": "persona.md",
    "chat_reply_model": "",
    "chat_models": {},
    "knowledge_dir": "data/raw",
    "vector_store_file": "data/knowledge_store.json",
    "memory_file": "data/memory.jsonl",
    "top_k": 4,
    "max_tool_rounds": 6,
    "temperature": 0.7,
    "web_port": 8000,
}


def _merge(base: dict, override: dict) -> dict:
    """浅合并两层配置：字典递归合并，其余直接覆盖。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> dict:
    """读取 config.json，缺省字段用 DEFAULTS 补齐。"""
    cfg = dict(DEFAULTS)
    p = path or (ROOT / "config.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _merge(cfg, user_cfg)
    return cfg


def resolve_path(cfg: dict, key: str, value: str) -> Path:
    """把配置里的相对路径解析为基于项目根目录的绝对路径。"""
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return p
