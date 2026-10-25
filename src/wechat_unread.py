"""未读红点识别（规则 5）。

在会话列表区域（左侧）用颜色检测微信的红色角标/红点，
返回红点中心坐标，供后续点击打开对应会话。
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_unread(
    img,
    list_x0: int = 55,
    list_x1: int = 320,
    y0: int = 80,
) -> list[dict]:
    """返回 [{"x","y","w","h","area"}]，按 y 从大到小（越靠下越新）或升序。"""
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    y0 = max(0, min(y0, h))
    x1 = min(list_x1, w)
    region = arr[y0:, list_x0:x1]
    if region.size == 0:
        return []

    r = region[:, :, 0].astype(np.int16)
    g = region[:, :, 1].astype(np.int16)
    b = region[:, :, 2].astype(np.int16)
    # 微信红点约 #FA5151 / #F5222D
    mask = (
        (r > 165)
        & (g < 95)
        & (b < 95)
        & (r - g > 80)
        & (r - b > 80)
    ).astype(np.uint8)

    # 带数字的徽标中间是白字，先做闭运算把断层补上
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    dots: list[dict] = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        # 真红点在头像右上角：实测 cx≈115、7x7、fill≈0.8
        if not (6 <= ww <= 22 and 6 <= hh <= 22):
            continue
        if not (30 <= area <= 400):
            continue
        fill = area / float(ww * hh)
        ratio = ww / float(hh)
        if not (0.7 <= ratio <= 1.5):
            continue
        # 小尺寸（真红点通常 7-10px）要求接近实心圆：
        # 群头像里的小红图块填充率低，会被这里过滤掉
        if ww < 12:
            if fill < 0.65 or not (0.85 <= ratio <= 1.15):
                continue
        else:
            # 带数字的大徽标/与红色头像粘连时，填充率会明显偏低，放宽阈值
            if fill < 0.35:
                continue
        cx = int(x + ww / 2 + list_x0)
        cy = int(y + hh / 2 + y0)
        if not (105 <= cx <= 125):
            continue
        # 红点必须在头像框右侧外侧（头像内部的红色图案不算；头像右边界约 111）
        if cx <= 111:
            continue
        dots.append(
            {
                "x": cx,
                "y": cy,
                "w": int(ww),
                "h": int(hh),
                "area": int(area),
            }
        )
    dots.sort(key=lambda d: d["y"])
    return dots
