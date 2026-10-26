"""抓取微信窗口并 OCR，用于校准坐标（消息区 / 输入框位置）。

用法：
    python scripts/wechat_capture.py [--title 微信] [--keep]

    --keep  保留截图（默认 OCR 完成后删除）

实现：
  1. 找到微信主窗口；
  2. 优先用 PrintWindow(PW_RENDERFULLCONTENT) 直接抓窗口（被遮挡也能抓）；
  3. 若抓到黑屏则回退：把窗口置前，用 ImageGrab 截该窗口区域；
  4. 保存 wechat_capture.png，并对整窗做 OCR，打印每个文本块的坐标。

输出坐标是相对窗口左上角的，后面写自动回复时用来定位消息区与输入框。
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageGrab
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "wechat_capture.png"

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


def capture_printwindow(hwnd: int) -> Image.Image | None:
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
    bmi.bmiHeader.biHeight = -h  # 自上而下
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0  # BI_RGB
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mfc_dc, bmp, 0, h, buf, ctypes.byref(bmi), 0)

    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(mfc_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)
    return img


def is_blank(img: Image.Image) -> bool:
    import numpy as np

    arr = np.asarray(img)
    return float(arr.std()) < 3.0


def capture_foreground(hwnd: int) -> Image.Image:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.6)
    left, top, w, h = window_rect(hwnd)
    return ImageGrab.grab(bbox=(left, top, left + w, top + h)).convert("RGB")


def main() -> int:
    title = "微信"
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]

    hwnd = find_wechat_hwnd(title)
    if not hwnd:
        print("找不到微信窗口，请确认微信已登录并打开主界面。")
        return 1
    print(f"微信窗口句柄：{hwnd}")
    print(f"窗口区域：{window_rect(hwnd)}")

    img = capture_printwindow(hwnd)
    mode = "PrintWindow"
    if img is None or is_blank(img):
        print("PrintWindow 抓到黑屏，回退到前台截屏（会短暂把微信置前）…")
        img = capture_foreground(hwnd)
        mode = "ImageGrab(foreground)"

    keep = "--keep" in sys.argv
    img.save(OUT)
    print(f"已保存：{OUT}（{img.size[0]}x{img.size[1]}，方式={mode}）")

    engine = RapidOCR()
    result, elapse = engine(str(OUT))
    print(f"OCR 用时：{elapse}，文本框数：{len(result) if result else 0}")
    if result:
        for item in result:
            box, text, score = item[0], item[1], item[2]
            score = float(score)
            if score < 0.5:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            print(f"[{score:.2f}] x={int(min(xs)):>4} y={int(min(ys)):>4} w={int(max(xs)-min(xs)):>4}  {text}")

    if not keep:
        try:
            OUT.unlink(missing_ok=True)
            print(f"已删除截图：{OUT.name}（加 --keep 可保留）")
        except Exception as e:
            print(f"[warn] 删除截图失败：{e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
