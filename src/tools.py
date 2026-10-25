"""工具集：让分身拥有「执行能力」。

默认所有文件操作都被限制在项目根目录内，禁止越界访问（防误删/隐私泄露）。
后续要加「执行命令行」等高权限工具时，请务必加白名单或二次确认。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------- #
# 工具的 OpenAI function-calling 声明（交给模型决定何时调用）
# ---------------------------------------------------------------------- #
TOOLS_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间。当用户询问时间、日期，或需要时间上下文时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "把一条重要的、长期有效的信息写入长期记忆（例如用户说过的偏好、事实、决定）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要记住的内容，用一句话清晰表述",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取项目目录内的文本文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对项目根目录的文件路径，例如 data/raw/example.md",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "在项目目录内写入一个文本文件（会覆盖同名文件）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对项目根目录的文件路径",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文本内容",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出项目目录内某个子目录下的文件，path 留空表示根目录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对项目根目录的目录路径，可留空",
                    }
                },
                "required": [],
            },
        },
    },
]


def _safe_path(path: str) -> Path:
    """把相对路径解析为项目根目录内的绝对路径，并禁止越界。"""
    p = (ROOT / path).resolve()
    root = ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError as e:
        raise ValueError(f"路径越界，禁止访问项目目录之外的位置：{path}") from e
    return p


# ---------------------------------------------------------------------- #
# 工具执行分发
# ---------------------------------------------------------------------- #
def execute(tool_call: dict, memory) -> str:
    """执行单个工具调用，返回字符串结果（会回传给模型）。"""
    func = tool_call.get("function", {})
    name = func.get("name", "")
    try:
        args = json.loads(func.get("arguments") or "{}")
    except json.JSONDecodeError:
        args = {}

    if name == "get_time":
        return time.strftime("%Y-%m-%d %H:%M:%S %A")

    if name == "remember":
        content = str(args.get("content", "")).strip()
        if not content:
            return "没有可记住的内容。"
        memory.append("memory", content)
        return "已写入长期记忆。"

    if name == "read_file":
        p = _safe_path(str(args.get("path", "")))
        if not p.is_file():
            return f"文件不存在：{p}"
        return p.read_text(encoding="utf-8", errors="replace")[:20000]

    if name == "write_file":
        p = _safe_path(str(args.get("path", "")))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(args.get("content", "")), encoding="utf-8")
        return f"已写入：{p}"

    if name == "list_dir":
        p = _safe_path(str(args.get("path", "")))
        if not p.is_dir():
            return f"目录不存在：{p}"
        entries = sorted(x.name for x in p.iterdir())
        return "\n".join(entries) if entries else "（空目录）"

    return f"未知工具：{name}"
