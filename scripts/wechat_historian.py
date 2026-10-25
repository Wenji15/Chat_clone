"""历史聊天提取（规则 3）：向上滚动逐屏 OCR，去重后写入 data/wechat/<会话>/。

用法：
    python scripts/wechat_historian.py --screens 5 --step 5

说明：
- 会滚动当前打开的聊天窗口，结束后自动滚回底部；
- 只采集当前会话，不做会话切换；
- 图片/表情/语音没有文字，会被跳过。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyautogui
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.wechat_store import append_messages, export_markdown  # noqa: E402
from src.wechat_win import (  # noqa: E402
    find_wechat_hwnd,
    set_dpi_aware,
    window_rect,
)
import wechat_screener as ws  # noqa: E402


def _group_lines(lines, gap: int = 24):
    """把同一气泡内换行的多行合并成一条消息。"""
    groups = []
    cur = []
    for y, t in sorted(lines):
        if cur and y - cur[-1][0] <= gap:
            cur.append((y, t))
        else:
            if cur:
                groups.append((cur[0][0], "".join(x for _y, x in cur)))
            cur = [(y, t)]
    if cur:
        groups.append((cur[0][0], "".join(x for _y, x in cur)))
    return groups


def screen_messages(engine, hwnd):
    got = ws.analyze_once(engine, hwnd)
    if got is None:
        return "", []
    _last_in, _last_out, incoming, outgoing, title = got
    items = [(y, "them", t) for y, t in _group_lines(incoming)]
    items += [(y, "me", t) for y, t in _group_lines(outgoing)]
    items.sort(key=lambda x: x[0])
    return title, [{"role": role, "text": text} for _y, role, text in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", type=int, default=5, help="向上翻几屏")
    ap.add_argument("--step", type=int, default=5, help="每屏滚轮格数")
    args = ap.parse_args()

    set_dpi_aware()
    pyautogui.FAILSAFE = True
    hwnd = find_wechat_hwnd()
    if not hwnd:
        print("找不到微信窗口")
        return 1

    left, top, w, h = window_rect(hwnd)
    # 点输入框聚焦（安全，不会打开图片/链接）
    pyautogui.click(left + int(w * 0.55), top + h - 90)
    time.sleep(0.4)
    # 鼠标移到消息区，滚轮才作用于聊天内容
    pyautogui.moveTo(left + int(w * 0.6), top + int(h * 0.4))

    engine = RapidOCR()
    screens: list[list[dict]] = []
    title = ""
    for i in range(args.screens):
        t, msgs = screen_messages(engine, hwnd)
        if not t:
            print(f"[{i}] 未读到会话标题，停止")
            break
        if title and t != title:
            print(f"[{i}] 会话变了（{title} -> {t}），停止")
            break
        title = t
        screens.append(msgs)
        print(f"[{i}] {title}：本屏 {len(msgs)} 条")
        pyautogui.scroll(args.step)
        time.sleep(0.8)

    if not title:
        print("没有采集到任何会话（可能微信没停在聊天窗口）")
        return 1

    # screens 是自底向上采集，反转成自顶向下；与已合并尾部做重叠去重
    merged: list[dict] = []
    for msgs in reversed(screens):
        for m in msgs:
            key = (m["role"], m["text"])
            if any((x["role"], x["text"]) == key for x in merged[-12:]):
                continue
            merged.append(m)

    added = append_messages(title, merged)
    md = export_markdown(title)
    print(f"\n采集 {len(merged)} 条，新增入库 {added} 条 -> {md}")

    pyautogui.scroll(-2000)
    time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
