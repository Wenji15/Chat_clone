"""从“聊天记录”窗口采集历史（比聊天框更适合：每条都带发送者和精确时间）。

前提：先在聊天右上角菜单打开“聊天记录”窗口（本脚本自动找到它）。
用法：
    python scripts/wechat_collect_history.py --screens 10 --step 1
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pyautogui
import pyperclip
from PIL import ImageGrab
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import wechat_screener as ws  # noqa: E402
from src.wechat_store import (  # noqa: E402
    append_messages,
    chat_dir,
    export_markdown,
    load_messages,
    safe_name,
    strip_parens,
)
from src.wechat_win import capture, find_wechat_hwnd, set_dpi_aware, window_rect  # noqa: E402

user32 = ctypes.windll.user32

HIST_TIME_RE = re.compile(
    r"^(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})$"
)
TITLE_RE = re.compile(r'"([^"]+)"的聊天记录')


def find_history_window() -> tuple[int, str]:
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(h, _l):
        if user32.IsWindowVisible(h):
            n = user32.GetWindowTextLengthW(h)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(h, buf, n + 1)
                if "聊天记录" in buf.value:
                    found.append((h, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else (0, "")


def parse_hist_time(text: str) -> str | None:
    t = (text or "").replace(" ", "").replace("：", ":")
    m = HIST_TIME_RE.match(t)
    if not m:
        return None
    y, mo, d, hh, mm = map(int, m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}"


def ocr_bands(engine, img, band_h: int = 700) -> list:
    """分段 OCR：避免大窗口被整体缩小导致小字丢失。"""
    w, h = img.size
    out: list = []
    y = 0
    while y < h:
        band = img.crop((0, y, w, min(h, y + band_h)))
        res = ws.ocr_image(engine, band)
        for it in res or []:
            box = it[0]
            newbox = [[p[0], p[1] + y] for p in box]
            out.append([newbox, it[1], it[2]])
        y += band_h
    return out


def parse_history(
    engine, img, chat_name: str, me_names: set, is_group: bool = False
) -> list[dict]:
    gray = img.convert("L")
    result = ocr_bands(engine, img)
    items = []
    for it in result:
        box, text, score = it[0], str(it[1]).strip(), float(it[2])
        if score < 0.5 or not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        if y0 < 140:  # 跳过标题/搜索框/标签栏
            continue
        items.append(
            {
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "dark": ws._darkness(gray, x0, y0, x1, y1 + 4),
                "ts": parse_hist_time(text),
            }
        )

    w = img.size[0]
    right_thr = w * 0.55
    left_thr = w * 0.45
    times = sorted(
        (it["y0"], it["ts"]) for it in items if it["ts"] and it["x0"] > right_thr
    )
    senders = sorted(
        (
            it
            for it in items
            if it["dark"] > 120 and not it["ts"] and it["x0"] < left_thr
        ),
        key=lambda i: i["y0"],
    )
    contents = [it for it in items if it["dark"] <= 120 and not it["ts"]]
    UI_NOISE = ("定位到聊天位置",)

    rows: list[dict] = []
    last_sender = ""
    last_ts = ""
    for it in sorted(contents, key=lambda i: i["y0"]):
        if any(n in it["text"] for n in UI_NOISE):
            continue
        # 忽略接龙/编号名单行
        if re.match(r"^\d{1,3}[.、]\s*\S", it["text"]) or "接龙" in it["text"]:
            continue
        # 忽略数字/符号碎片（长度<=2）
        t = it["text"].strip()
        if len(t) <= 2 and re.fullmatch(r"[\d\W_]+", t):
            continue
        cand = [
            s
            for s in senders
            if -5 <= it["y0"] - s["y0"] <= 60 and abs(it["x0"] - s["x0"]) < 90
        ]
        if cand:
            s = max(cand, key=lambda x: x["y0"])
            sender = s["text"]
            sy = s["y0"]
        elif last_sender:
            sender = last_sender
            sy = it["y0"]
        elif is_group:
            # 群聊里没有发送者标签（多半是接龙/系统内容），直接丢弃
            continue
        else:
            sender = chat_name
            sy = it["y0"]
        if sender not in me_names and sender != "我":
            last_sender = sender
        tcand = [t for t in times if abs(t[0] - sy) <= 30]
        if tcand:
            ts = min(tcand, key=lambda t: abs(t[0] - sy))[1]
            last_ts = ts
        else:
            ts = last_ts  # 同一分钟的消息不重复显示时间，继承上一条
        role = "me" if sender in me_names or sender == "我" else "them"
        rows.append(
            {
                "role": role,
                "sender": sender,
                "text": it["text"],
                "ts": ts,
                "y": it["y0"],
            }
        )

    # 合并同一行内的多行内容
    merged: list[dict] = []
    for r in rows:
        if (
            merged
            and merged[-1]["role"] == r["role"]
            and merged[-1]["sender"] == r["sender"]
            and r["y"] - merged[-1]["y"] <= 24
        ):
            merged[-1]["text"] += r["text"]
        else:
            merged.append(dict(r))
    return [
        {"role": m["role"], "sender": m["sender"], "text": m["text"], "ts": m["ts"]}
        for m in merged
    ]


def open_chat(hwnd: int, name: str) -> None:
    """用微信搜索框打开指定会话。"""
    left, top, w, h = window_rect(hwnd)
    pyautogui.click(left + 180, top + 46)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "a")
    pyperclip.copy(name)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.8)
    pyautogui.press("enter")
    time.sleep(1.2)


def open_history_window(hwnd: int, engine: RapidOCR) -> bool:
    """点聊天右上角“...”，在菜单里点“聊天记录”。"""
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)
    left, top, w, h = window_rect(hwnd)
    for dx in (120, 110, 130):
        x, y = left + w - dx, top + 55
        pyautogui.click(x, y)
        time.sleep(0.3)
        pyautogui.click(x, y)
        time.sleep(0.8)
        img = ImageGrab.grab()
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        res, _ = engine(arr)
        hit = None
        for it in res or []:
            text = str(it[1])
            if "聊天记录" in text and float(it[2]) >= 0.5:
                xs = [p[0] for p in it[0]]
                ys = [p[1] for p in it[0]]
                hit = (int(sum(xs) / 4), int(sum(ys) / 4))
                break
        if hit:
            pyautogui.click(hit[0], hit[1])
            time.sleep(1.2)
            return True
        pyautogui.press("esc")
        time.sleep(0.3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", type=int, default=10)
    ap.add_argument(
        "--step",
        type=int,
        default=10,
        help="每屏滚动格数(4-20；1 格约 3 行，10 格约 600px，留重叠)",
    )
    ap.add_argument("--name", type=str, default="")
    ap.add_argument("--chat", type=str, default="", help="先搜索并打开这个会话")
    ap.add_argument("--auto-open", action="store_true", help="自动打开聊天记录窗口")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="先备份并清空该会话数据，再重新采集（修复顺序）",
    )
    args = ap.parse_args()
    args.step = max(4, min(20, args.step))

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    set_dpi_aware()
    pyautogui.FAILSAFE = True

    hwnd, title = find_history_window()
    if not hwnd:
        main_hwnd = find_wechat_hwnd()
        if main_hwnd and args.chat:
            print(f"打开会话：{args.chat}")
            open_chat(main_hwnd, args.chat)
            time.sleep(1.0)
        if main_hwnd and args.auto_open:
            print("尝试自动打开聊天记录窗口…")
            auto_engine = RapidOCR()
            if open_history_window(main_hwnd, auto_engine):
                hwnd, title = find_history_window()
    if not hwnd:
        print("未找到“聊天记录”窗口，请先从聊天右上角菜单打开它。")
        return 1

    m = TITLE_RE.search(title)
    raw_name = m.group(1) if m else title
    is_group = bool(re.search(r"[（(]\d+[)）]", title))
    chat_name = args.name or strip_parens(raw_name)
    try:
        aliases = json.loads((ROOT / "chat_aliases.json").read_text(encoding="utf-8"))
        chat_name = aliases.get(chat_name, chat_name)
    except Exception:
        pass
    rules: dict = {}
    try:
        rules = json.loads((ROOT / "wechat_rules.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    me_names = set(rules.get("my_names") or ["我"])
    me_display = "、".join(sorted(me_names))
    print(f"聊天记录窗口：{title} -> 存储为 {chat_name}（我方：{me_display}）")

    orig = window_rect(hwnd)
    screen_h = user32.GetSystemMetrics(1)
    user32.MoveWindow(hwnd, orig[0], 0, orig[2], screen_h - 40, True)  # 只拉高，宽度不变
    time.sleep(0.8)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    left, top, w, h = window_rect(hwnd)
    pyautogui.click(left + int(w * 0.5), top + 12)  # 点标题栏聚焦
    time.sleep(0.5)
    pyautogui.moveTo(left + int(w * 0.5), top + int(h * 0.5))
    pyautogui.press("home")
    pyautogui.scroll(3000)  # 回顶部
    time.sleep(0.8)

    engine = RapidOCR()
    screens: list[list[dict]] = []
    seen: set[tuple[str, str]] = set()
    no_new = 0
    for i in range(args.screens):
        if (ROOT / "STOP_COLLECT").exists():
            try:
                (ROOT / "STOP_COLLECT").unlink()
            except OSError:
                pass
            print("收到停止采集指令")
            break
        while (ROOT / "PAUSE_COLLECT").exists():
            if (ROOT / "STOP_COLLECT").exists():
                break
            time.sleep(0.3)
        msgs = parse_history(engine, capture(hwnd), chat_name, me_names, is_group)
        new = [
            m
            for m in msgs
            if (m.get("sender"), m.get("text")) not in seen
        ]
        seen.update((m.get("sender"), m.get("text")) for m in msgs)
        screens.append(msgs)
        print(f"[{i}] {len(msgs)} 条（新增 {len(new)}）")
        if i > 0 and not new:
            no_new += 1
            if no_new >= 2:
                print("连续两屏没有新消息，停止（可能已到底部）")
                break
            print("本屏无新消息，用 PageDown 兜底")
            pyautogui.press("pagedown")
            time.sleep(1.0)
            continue
        no_new = 0
        # 确保窗口在前台、鼠标在消息列表上，滚轮事件才会送到列表
        try:
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        time.sleep(0.1)
        pyautogui.moveTo(left + int(w * 0.5), top + int(h * 0.6))
        for _ in range(args.step):
            pyautogui.scroll(-1)
            time.sleep(0.04)
        time.sleep(0.8)

    if not any(screens):
        print("没有采集到任何消息")
        return 1

    # 聊天记录窗口是“最新在上、最旧在下”，从上往下滚是越来越旧：
    # 屏幕顺序要反，每屏内部也要反，才是时间正序
    all_msgs: list[dict] = []
    for msgs in reversed(screens):
        for m in reversed(msgs):
            if all_msgs and all_msgs[-1]["role"] == m["role"] and all_msgs[-1]["text"] == m["text"]:
                continue
            all_msgs.append(m)

    if args.rebuild:
        d = chat_dir(chat_name)
        p = d / "messages.jsonl"
        if p.exists():
            bak = d / f"messages.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            p.replace(bak)
            print(f"[rebuild] 旧数据已备份：{bak.name}")

    added = append_messages(chat_name, all_msgs)
    md = export_markdown(chat_name)
    print(f"\n采集 {len(all_msgs)} 条，新增入库 {added} 条 -> {md}")

    if not args.no_raw:
        raw = ROOT / "data" / "raw" / "wechat" / f"{safe_name(chat_name)}.md"
        raw.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {chat_name} 聊天记录", ""]
        # 原料按窗口习惯导出：新消息在上
        for m in reversed(load_messages(chat_name)):
            lines.append(f"- [{m.get('ts', '')}] {m.get('sender', '')}：{m.get('text', '')}")
        raw.write_text("\n".join(lines), encoding="utf-8")
        print(f"已导出知识库原料：{raw}")

    pyautogui.scroll(-3000)
    user32.MoveWindow(hwnd, orig[0], orig[1], orig[2], orig[3], True)  # 还原
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
