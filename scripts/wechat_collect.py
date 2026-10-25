"""历史聊天采集：向上滚动逐屏 OCR，还原“发送者 / 内容 / 时间”，按会话分库保存。

规则对齐：
- 单聊：一个联系人一个目录；群聊：一个群一个目录；
- 每条记录标注 sender / text / ts，写入 data/wechat/<会话>/messages.jsonl；
- 同时导出 data/raw/wechat/<会话>.md，供 ingest.py 入库。

用法：
    python scripts/wechat_collect.py --screens 10 --step 3
    python scripts/wechat_collect.py --screens 20 --step 3 --name 11
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyautogui
import pyperclip
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import wechat_screener as ws  # noqa: E402
from src.wechat_store import (  # noqa: E402
    append_messages,
    export_markdown,
    load_messages,
    safe_name,
    strip_parens,
)
from src.wechat_win import capture, find_wechat_hwnd, set_dpi_aware, window_rect  # noqa: E402
from src.wechat_layout import detect_chat_x0  # noqa: E402

TIME_RE = re.compile(
    r"^(?:(今天|昨天|前天|星期[一二三四五六日])\s*)?(\d{1,2})[:：](\d{2})$"
)
DATE_RE = re.compile(
    r"^(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?\s*(\d{1,2})[:：](\d{2})$"
)
WEEKDAYS = {
    "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3,
    "星期五": 4, "星期六": 5, "星期日": 6, "星期天": 6,
}


def parse_time(text: str, now: datetime) -> str | None:
    t = (text or "").replace("：", ":").strip()
    m = DATE_RE.match(t)
    if m:
        y, mo, d, hh, mm = map(int, m.groups())
        return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}"
    m = TIME_RE.match(t)
    if not m:
        return None
    day, hh, mm = m.groups()
    hh, mm = int(hh), int(mm)
    base = now
    if day == "昨天":
        base = now - timedelta(days=1)
    elif day == "前天":
        base = now - timedelta(days=2)
    elif day and day in WEEKDAYS:
        delta = (now.weekday() - WEEKDAYS[day]) % 7
        base = now - timedelta(days=delta)
    dt = base.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if day is None and dt > now:
        dt -= timedelta(days=1)
    return dt.strftime("%Y-%m-%d %H:%M")


def parse_screen(
    engine, img, chat_name: str, now: datetime, me_name: str = "我"
) -> list[dict]:
    """解析一屏聊天区，返回 [{role,sender,text,ts}]（按屏幕内从上到下）。"""
    w, h = img.size
    x0 = detect_chat_x0(img, 315)
    y0 = min(60, h // 3)
    x1 = max(x0 + 60, w - 15)
    y1 = max(y0 + 50, h - 165)
    crop = img.crop((x0, y0, x1, y1))
    result = ws.ocr_image(engine, crop)
    gray = crop.convert("L")
    rgb = np.asarray(crop.convert("RGB"))
    crop_w = crop.size[0]

    items = []
    for it in result:
        box, text, score = it[0], str(it[1]).strip(), float(it[2])
        if score < 0.5 or not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        ix0, iy0, ix1, iy1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        items.append(
            {
                "text": text,
                "x0": ix0,
                "y0": iy0,
                "x1": ix1,
                "y1": iy1,
                "cx": (ix0 + ix1) // 2,
                "dark": ws._darkness(gray, ix0, iy0, ix1, iy1 + 4),
                "ts": parse_time(text, now),
            }
        )

    # 时间分隔符（居中显示）
    stamps = sorted(
        (it["y0"], it["ts"])
        for it in items
        if it["ts"] and abs(it["cx"] - crop_w / 2) < crop_w * 0.28
    )

    def ts_for(y: int) -> str:
        if not stamps:
            return now.strftime("%Y-%m-%d %H:%M")
        _sy, st = min(stamps, key=lambda s: abs(s[0] - y))
        return st

    labels = sorted(
        (it for it in items if it["dark"] > 120 and not it["ts"]),
        key=lambda i: i["y0"],
    )

    is_group = bool(labels)

    def sender_for(y: int, x0_: int) -> str:
        cand = [
            l
            for l in labels
            if -5 <= y - l["y0"] <= 60 and abs(l["x0"] - x0_) < 110
        ]
        if cand:
            return max(cand, key=lambda l: l["y0"])["text"]
        return "未知" if is_group else chat_name

    items.sort(key=lambda i: i["y0"])
    msgs = []
    last_sender = ""
    for it in items:
        if it["dark"] > 120 or it["ts"]:
            continue
        if not ws._meaningful(it["text"]):
            continue
        green = ws._green_fraction(rgb, it["x0"], it["y0"], it["x1"], it["y1"]) > 0.2
        role = "me" if green else "them"
        if role == "me":
            sender = me_name
        else:
            sender = sender_for(it["y0"], it["x0"])
            if sender == "未知" and last_sender:
                sender = last_sender  # 同一人连发时名字只出现一次
            last_sender = sender
        msgs.append(
            {
                "role": role,
                "sender": sender,
                "text": it["text"],
                "ts": ts_for(it["y0"]),
                "y": it["y0"],
            }
        )

    # 合并同一气泡内的多行
    msgs.sort(key=lambda m: m["y"])
    merged: list[dict] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"] and m["y"] - merged[-1]["y"] <= 24:
            merged[-1]["text"] += m["text"]
        else:
            merged.append(dict(m))
    return [
        {"role": m["role"], "sender": m["sender"], "text": m["text"], "ts": m["ts"]}
        for m in merged
    ]


def similar(a: str, b: str, ratio: float = 0.8) -> bool:
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def append_screen(merged: list[dict], msgs: list[dict]) -> None:
    """把新的一屏接到已合并结果后，去掉与尾部重叠的部分。"""
    k = min(len(merged), len(msgs), 15)
    best = 0
    for kk in range(k, 0, -1):
        if all(
            merged[-kk + i]["role"] == msgs[i]["role"]
            and similar(merged[-kk + i]["text"], msgs[i]["text"])
            for i in range(kk)
        ):
            best = kk
            break
    merged.extend(msgs[best:])


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", type=int, default=10, help="向上翻几屏")
    ap.add_argument("--step", type=int, default=1, help="每屏翻几页(1-3)")
    ap.add_argument("--chat", type=str, default="", help="先搜索并打开这个会话")
    ap.add_argument("--name", type=str, default="", help="会话名（覆盖 OCR 标题）")
    ap.add_argument("--no-raw", action="store_true", help="不导出 data/raw")
    args = ap.parse_args()
    args.step = max(1, min(3, args.step))

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    set_dpi_aware()
    pyautogui.FAILSAFE = True
    hwnd = find_wechat_hwnd()
    if not hwnd:
        print("找不到微信窗口")
        return 1

    engine = RapidOCR()
    if args.chat:
        print(f"打开会话：{args.chat}")
        open_chat(hwnd, args.chat)
        time.sleep(1.0)

    title = args.name or ws.read_title(engine, capture(hwnd)) or ""
    if not title:
        print("未检测到聊天窗口：请先在微信打开目标会话，或在“采集会话”里填会话名。")
        return 1
    if not args.name:
        title = strip_parens(title)

    # 会话别名：把 OCR 标题归一到统一的存储名（如 11(18) -> 🐦🐦）
    if not args.name:
        try:
            aliases = json.loads((ROOT / "chat_aliases.json").read_text(encoding="utf-8"))
            title = aliases.get(title, title)
        except Exception:
            pass

    left, top, w, h = window_rect(hwnd)
    pyautogui.click(left + int(w * 0.55), top + h - 90)
    time.sleep(0.3)
    pyautogui.click(left + 420, top + int(h * 0.5))  # 聚焦消息区
    time.sleep(0.3)
    pyautogui.press("end")  # 先到最底部
    time.sleep(0.8)

    rules: dict = {}
    try:
        rules = json.loads((ROOT / "wechat_rules.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    me_name = (rules.get("my_names") or ["我"])[0]
    print(f"会话：{title}（我方署名：{me_name}）")

    screens: list[list[dict]] = []
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
        if (ROOT / "STOP_COLLECT").exists():
            try:
                (ROOT / "STOP_COLLECT").unlink()
            except OSError:
                pass
            print("收到停止采集指令")
            break
        msgs = parse_screen(engine, capture(hwnd), title, datetime.now(), me_name)
        screens.append(msgs)
        print(f"[{i}] {len(msgs)} 条")
        for _ in range(args.step):
            pyautogui.press("pageup")
        time.sleep(0.7)

    if not any(screens):
        print("没有采集到任何消息（聊天可能是空的，或窗口不在聊天区）")
        return 1

    merged: list[dict] = []
    for msgs in reversed(screens):
        append_screen(merged, msgs)

    added = append_messages(title, merged)
    md = export_markdown(title)
    print(f"\n采集 {len(merged)} 条，新增入库 {added} 条 -> {md}")

    if not args.no_raw:
        raw = ROOT / "data" / "raw" / "wechat" / f"{safe_name(title)}.md"
        raw.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {title} 聊天记录", ""]
        for m in load_messages(title):
            lines.append(f"- [{m.get('ts', '')}] {m.get('sender', '')}：{m.get('text', '')}")
        raw.write_text("\n".join(lines), encoding="utf-8")
        print(f"已导出知识库原料：{raw}")

    pyautogui.press("end")
    time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
