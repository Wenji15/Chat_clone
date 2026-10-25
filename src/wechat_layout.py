"""微信窗口版面自动校准。

通过灰度列均值检测“会话列表 / 聊天区”的分隔线，避免写死 315。
适配：窗口缩放、拖动左右分隔条（浅色主题）。
"""
from __future__ import annotations

import numpy as np

DEFAULT_CHAT_X0 = 315


def detect_chat_x0_ex(img, default: int = DEFAULT_CHAT_X0) -> tuple[int, bool]:
    """返回 (聊天区左边界, 是否真的检测到分隔线)。"""
    try:
        arr = np.asarray(img.convert("L")).astype(int)
    except Exception:
        return default, False
    h, w = arr.shape
    if h < 300 or w < 400:
        return default, False

    y0, y1 = 100, max(200, h - 150)
    col = arr[y0:y1].mean(axis=0)

    lo = max(120, int(w * 0.2))
    hi = min(int(w * 0.85), w - 50)
    for x in range(lo, hi):
        before = col[max(0, x - 6) : x].mean()
        after = col[x : x + 6].mean()
        # 聊天区背景比列表区亮（约 250 vs 235）
        if after - before >= 8 and after > 245:
            return x, True
    return min(default, int(w * 0.45)), False


def detect_chat_x0(img, default: int = DEFAULT_CHAT_X0) -> int:
    """返回聊天区左边界（相对窗口的 x 像素）。"""
    return detect_chat_x0_ex(img, default)[0]
