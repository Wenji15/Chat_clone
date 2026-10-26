"""向指定会话连续发送 N 条测试消息（最多 20 条，带停止标志）。

用法：
    python scripts/wechat_burst.py --chat 文件传输助手 --count 5 --text "测试 {i}"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pyautogui  # noqa: E402
from rapidocr_onnxruntime import RapidOCR  # noqa: E402

import wechat_screener as ws  # noqa: E402
from wechat_multi import list_titles  # noqa: E402
from src.agent import CloneAgent  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.wechat_win import (  # noqa: E402
    capture,
    find_wechat_hwnd,
    set_dpi_aware,
    window_rect,
)

MAX_COUNT = 20
STOP = ROOT / "STOP_BURST"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", required=True)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--text", default="测试 {i}")
    ap.add_argument(
        "--generate",
        action="store_true",
        help="用对话模型生成内容（根据当前窗口），而不是固定文本",
    )
    args = ap.parse_args()

    count = max(1, min(MAX_COUNT, args.count))
    if count != args.count:
        print(f"[burst] 条数已限制为 {count}（最多 {MAX_COUNT}）")

    set_dpi_aware()
    hwnd = find_wechat_hwnd()
    if not hwnd:
        print("[burst] 没找到微信窗口")
        return 1

    engine = RapidOCR()
    img = capture(hwnd)
    title = ws.read_title(engine, img)
    if args.chat not in title:
        rows = list_titles(engine, img, {args.chat})
        row = next(
            ((t, y) for t, y in rows if args.chat in t or t in args.chat),
            None,
        )
        if not row:
            print(f"[burst] 列表里找不到会话：{args.chat!r}")
            return 1
        left, top, _w, _h = window_rect(hwnd)
        pyautogui.click(left + 200, top + row[1])
        time.sleep(1.5)
        title = ws.read_title(engine, capture(hwnd))
        if args.chat not in title:
            print(f"[burst] 打开后标题不匹配：{title!r}")
            return 1
        print(f"[burst] 已打开：{title!r}")

    STOP.unlink(missing_ok=True)
    agent = None
    if args.generate:
        agent = CloneAgent(load_config(), ROOT)
        try:
            agent.llm.chat_native(
                [{"role": "user", "content": "ping"}], temperature=0.0
            )
            print("[burst] 回复模型已加载")
        except Exception as e:
            print(f"[burst] 模型预热失败：{e}")

    for i in range(1, count + 1):
        if STOP.exists():
            STOP.unlink(missing_ok=True)
            print("[burst] 收到停止")
            break
        if args.generate and agent:
            got = ws.analyze_once(engine, hwnd)
            msg = ""
            if got:
                _li, _lo, incoming, outgoing, _t = got
                msg = (
                    agent.chat_reply(ws.build_history(incoming, outgoing), chat=args.chat)
                    or ""
                )
            if not msg:
                print(
                    f"[burst] 模型返回空（{agent.last_skip_reason or '无'}），停止"
                )
                break
        else:
            msg = args.text.replace("{i}", str(i))
        ws.send_reply(hwnd, msg)
        print(f"[burst] {i}/{count} -> {msg!r}")
        time.sleep(max(0.5, args.interval))
    print("[burst] 结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
