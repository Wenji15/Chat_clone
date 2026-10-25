"""微信窗口抓取工具（ctypes + PrintWindow，无第三方窗口库）。

只负责“看”：定位微信主窗口、抓取窗口位图。不涉及任何输入/发送。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def set_dpi_aware() -> None:
    """让本进程按物理像素工作，避免 150% 缩放导致坐标错位。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def find_wechat_hwnd(title: str = "微信") -> int:
    hwnd = user32.FindWindowW(None, title)
    if hwnd:
        return hwnd
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd_, _lparam):
        if not user32.IsWindowVisible(hwnd_):
            return True
        length = user32.GetWindowTextLengthW(hwnd_)
        if length:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd_, buf, length + 1)
            if "微信" in buf.value or "WeChat" in buf.value:
                found.append(hwnd_)
        return True

    user32.EnumWindows(_enum, 0)
    return found[0] if found else 0


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def capture(hwnd: int) -> Image.Image | None:
    """用 PrintWindow 抓窗口（被遮挡也可），失败返回 None。"""
    left, top, w, h = window_rect(hwnd)
    if w <= 0 or h <= 0:
        return None
    hwnd_dc = user32.GetWindowDC(hwnd)
    mfc_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    gdi32.SelectObject(mfc_dc, bmp)
    user32.PrintWindow(hwnd, mfc_dc, 2)  # PW_RENDERFULLCONTENT

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mfc_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)

    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mfc_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)
    return img


def is_blank(img: Image.Image) -> bool:
    import numpy as np

    return float(np.asarray(img).std()) < 3.0


def bring_to_front(hwnd: int) -> None:
    """把窗口置前（发送前调用）。"""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
