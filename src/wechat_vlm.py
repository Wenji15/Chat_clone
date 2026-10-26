"""用本地视觉模型读取微信列表行里的会话名。

红点位置仍由颜色规则检测（比大模型判断更稳）；
这里只负责“这个红点对应哪一行、行里的会话名是什么”，
解决 OCR 把时间/预览误当会话名的问题。
"""
from __future__ import annotations

import base64
import hashlib
import io

from PIL import Image

from .ollama_client import OllamaClient

_CACHE: dict[str, str] = {}


def read_row_title(
    client: OllamaClient,
    img: Image.Image,
    dot_y: int,
    x0: int,
    model: str,
) -> str:
    """截取红点所在行（头像+标题+预览），让视觉模型读出会话名。"""
    # 只截标题那一行（约 33px 高），避免把上一行预览/下一行标题带进来
    crop = img.crop(
        (50, max(0, dot_y - 15), max(160, x0 - 100), min(img.height, dot_y + 18))
    )
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    key = hashlib.md5(b64.encode("ascii")).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    prompt = (
        "这是微信会话列表中的一行截图（包含头像、会话名、时间、最后一条消息预览）。"
        "请只输出最上方的会话名称；会话名可能是 emoji 或特殊符号，也要原样输出，"
        "不要输出时间、预览文字、标点或解释。"
        "如果看不清，只输出“未知”。"
    )
    text = client.chat_vision(prompt, [b64], model=model)
    line = ""
    if text and text.strip():
        line = text.strip().splitlines()[0].strip(" ：:·.-")
    _CACHE[key] = line
    return line
