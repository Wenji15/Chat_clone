"""微信屏幕监听 → 分身自动回复（独立运行版）。

不依赖 Web 服务：直接在本进程内加载 CloneAgent 生成回复。
只需要：① Ollama 在跑；② 微信登录并停在目标会话。

用法：
    python scripts/wechat_screener.py --once                 # 单次干跑（识别+拟回复，不发送）
    python scripts/wechat_screener.py --once --live           # 单次真发
    python scripts/wechat_screener.py --live --target 杨文祥  # 循环，持续自动回复
    python scripts/wechat_screener.py --live --target 杨文祥 --interval 3 --log logs/wechat_bot.log

停止方式：
    - 在项目根目录放一个名为 STOP 的文件（脚本会自动删除并退出）
    - 或在控制台按 Ctrl+C
    - 或把鼠标猛移到屏幕左上角（pyautogui FAILSAFE）

安全阀：目标会话锁、只回对方消息、模糊去重、冷却、启动首轮不回旧消息。
"""
from __future__ import annotations

import argparse
import json
import difflib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyautogui
import pyperclip
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import CloneAgent  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.wechat_win import (  # noqa: E402
    bring_to_front,
    capture,
    find_wechat_hwnd,
    is_blank,
    set_dpi_aware,
    window_rect,
)
from src.wechat_layout import detect_chat_x0  # noqa: E402

# --------------------------------------------------------------------- #
# 版面坐标（固定像素 + 从边缘锚定，抗窗口拉伸）
# --------------------------------------------------------------------- #
CHAT_X0_PX = 315
CHAT_X1_MARGIN = 15
MSG_TOP_PX = 60
MSG_BOTTOM_MARGIN = 165
INPUT_X_FRAC = 0.55
INPUT_Y_MARGIN = 90

TIME_RE = re.compile(
    r"^(昨天|前天|今天|星期[一二三四五六日])?\s*"
    r"(\d{1,2}[:：]\d{2}|\d{4}/\d{1,2}/\d{1,2}.*|\d{1,2}月\d{1,2}日.*)$"
)
NAME_GRAY_MIN = 120

# --------------------------------------------------------------------- #
# 分身（进程内单例）
# --------------------------------------------------------------------- #
_agent: CloneAgent | None = None


def get_agent() -> CloneAgent:
    global _agent
    if _agent is None:
        _agent = CloneAgent(load_config(), ROOT)
    return _agent


# --------------------------------------------------------------------- #
# OCR 与解析
# --------------------------------------------------------------------- #
def ocr_image(engine: RapidOCR, img: Image.Image) -> list:
    arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    result, _ = engine(arr)
    return result or []


def read_title(engine: RapidOCR, img: Image.Image) -> str:
    w, h = img.size
    x0 = detect_chat_x0(img, CHAT_X0_PX)
    strip = img.crop((max(0, x0 - 10), 35, min(w - 60, x0 + 420), min(78, h)))
    result = ocr_image(engine, strip)
    return "".join(str(it[1]).strip() for it in result if float(it[2]) >= 0.6)


def _darkness(gray, x0: int, y0: int, x1: int, y1: int) -> float:
    reg = np.asarray(gray)[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if reg.size == 0:
        return 255.0
    vals = np.sort(reg.ravel())
    k = max(1, int(len(vals) * 0.1))
    return float(vals[:k].mean())


def _meaningful(text: str) -> bool:
    # 至少含 1 个有效字符（汉字/字母/数字），单字消息如“6”“好”“嗯”也算
    return len(re.sub(r"[\W_]", "", text)) >= 1


def _green_fraction(rgb, x0: int, y0: int, x1: int, y1: int) -> float:
    """判断文字框周边是否为我方绿色气泡（微信我方气泡约 #95EC69）。"""
    h, w = rgb.shape[:2]
    xa = max(0, x0 - 25)
    xb = min(w, x1 + 25)
    ya = max(0, y0 - 2)
    yb = min(h, y1 + 4)
    if xb <= xa or yb <= ya:
        return 0.0
    reg = rgb[ya:yb, xa:xb].astype(int)
    r, g, b = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
    green = (g > 180) & (r > 100) & (r < 210) & (b < 160) & (g > r + 20) & (g > b + 20)
    return float(green.mean())


def split_messages(result: list, crop_w: int, gray, rgb):
    incoming: list[tuple[int, str]] = []
    outgoing: list[tuple[int, str]] = []
    for item in result:
        box, text, score = item[0], str(item[1]), float(item[2])
        if score < 0.5:
            continue
        t = text.strip()
        if not t or TIME_RE.match(t) or not _meaningful(t):
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        if y1 - y0 < 8:
            continue
        if _darkness(gray, x0, y0, x1, y1 + 4) > NAME_GRAY_MIN:
            continue  # 灰色名字标签
        if _green_fraction(rgb, x0, y0, x1, y1) > 0.2:
            outgoing.append((y0, t))  # 绿气泡优先判为我方（修复换行误判）
        elif (x0 + x1) / 2 < crop_w * 0.5:
            incoming.append((y0, t))
        else:
            outgoing.append((y0, t))
    return incoming, outgoing


def last_bubble(lines: list[tuple[int, str]], y_tol: int = 60) -> tuple[int, str] | None:
    if not lines:
        return None
    lines = sorted(lines, key=lambda t: t[0])
    y_bottom, _ = lines[-1]
    group = [t for t in lines if y_bottom - t[0] <= y_tol]
    group.sort(key=lambda t: t[0])
    return y_bottom, "\n".join(t[1] for t in group)


def build_history(incoming, outgoing, limit: int = 40) -> list[dict]:
    items = [(y, "them", t) for y, t in incoming] + [(y, "me", t) for y, t in outgoing]
    items.sort(key=lambda x: x[0])
    return [{"role": role, "text": text} for _y, role, text in items[-limit:]]


# --------------------------------------------------------------------- #
# 抓取 + 分析
# --------------------------------------------------------------------- #
def analyze_once(engine: RapidOCR, hwnd: int):
    img = capture(hwnd)
    if img is None or is_blank(img):
        return None
    w, h = img.size
    x0 = detect_chat_x0(img, CHAT_X0_PX)
    y0 = min(MSG_TOP_PX, h // 3)
    x1 = w - CHAT_X1_MARGIN
    y1 = max(y0 + 50, h - MSG_BOTTOM_MARGIN)
    crop = img.crop((x0, y0, x1, y1))
    result = ocr_image(engine, crop)
    incoming, outgoing = split_messages(
        result, crop.size[0], crop.convert("L"), np.asarray(crop.convert("RGB"))
    )
    title = read_title(engine, img)
    return last_bubble(incoming), last_bubble(outgoing), incoming, outgoing, title


def send_reply(hwnd: int, reply: str) -> None:
    pyautogui.FAILSAFE = True
    bring_to_front(hwnd)
    time.sleep(0.4)
    left, top, w, h = window_rect(hwnd)
    pyautogui.click(left + int(w * INPUT_X_FRAC), top + (h - INPUT_Y_MARGIN))
    time.sleep(0.2)
    pyperclip.copy(reply)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.3)
    pyautogui.press("enter")
    time.sleep(0.3)


# --------------------------------------------------------------------- #
# 日志与停止
# --------------------------------------------------------------------- #
class Logger:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def stop_requested() -> bool:
    p = ROOT / "STOP"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass
        return True
    return False


# --------------------------------------------------------------------- #
def load_bot_config() -> dict:
    p = ROOT / "bot_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] 读取 bot_config.json 失败：{e}")
    return {}


def load_state() -> dict:
    p = ROOT / "data" / "bot_state.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(d: dict) -> None:
    p = ROOT / "data" / "bot_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", default=None, help="真正发送")
    ap.add_argument("--once", action="store_true", help="只跑一次分析")
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--log", type=str, default=None)
    ap.add_argument("--reply-existing", action="store_true", default=None, help="启动时直接回复当前最后一条对方消息")
    args = ap.parse_args()

    bc = load_bot_config()
    if args.live is None:
        # --once 默认干跑；只有显式 --live 才发
        args.live = False if args.once else bool(bc.get("live", False))
    if args.interval is None:
        args.interval = float(bc.get("interval", 3.0))
    if args.cooldown is None:
        args.cooldown = float(bc.get("cooldown", 6.0))
    if args.target is None:
        args.target = str(bc.get("target", ""))
    if args.log is None:
        args.log = str(bc.get("log", ""))
    if args.reply_existing is None:
        args.reply_existing = bool(bc.get("reply_existing", False))

    log = Logger(Path(args.log) if args.log else None)
    set_dpi_aware()
    pyautogui.FAILSAFE = True

    log(f"启动：mode={'LIVE' if args.live else 'DRY-RUN'} target={args.target!r} interval={args.interval}")
    engine = RapidOCR()

    state = load_state()
    last_seen: str | None = state.get("last_replied")
    last_reply_time = 0.0
    initialised = False
    hwnd = 0

    while True:
        if stop_requested():
            log("收到 STOP，退出。")
            return 0

        if not hwnd or not user32_is_window(hwnd):
            hwnd = find_wechat_hwnd()
            if not hwnd:
                log("[wait] 未找到微信窗口，5 秒后重试…")
                time.sleep(5)
                continue

        try:
            got = analyze_once(engine, hwnd)
        except Exception as e:  # 抓取/OCR 异常不崩
            log(f"[error] 分析失败：{e}")
            time.sleep(args.interval)
            continue

        if got is None:
            log("[warn] 抓取失败或黑屏，跳过")
            time.sleep(args.interval)
            continue

        last_in, last_out, incoming, outgoing, title = got

        if args.target and args.target not in title:
            if args.once:
                log(f"当前会话 {title!r} ≠ 目标 {args.target!r}，跳过")
                return 0
            time.sleep(args.interval)
            continue

        should_reply = last_in is not None and (last_out is None or last_in[0] > last_out[0])

        if args.once:
            log(f"会话标题：{title!r}")
            log(f"对方：{incoming}")
            log(f"我方：{outgoing}")
            if should_reply:
                log(f"→ 触发，消息：{last_in[1]!r}")
                reply = get_agent().chat_reply(build_history(incoming, outgoing), chat=title)
                log(f"→ 回复：{reply!r}")
                if not reply:
                    log("[skip] 空回复，不发送")
                elif args.live:
                    send_reply(hwnd, reply)
                    log("[sent] 已发送")
                    save_state({"last_replied": last_in[1], "ts": time.time()})
                else:
                    log("[dry-run] 未发送")
            else:
                log("→ 最后一条不是对方发的，不回复")
            return 0

        if not initialised:
            initialised = True
            prev = (state.get("last_replied") or "").strip()
            cur = last_in[1] if last_in else None
            if args.reply_existing:
                log("[init] reply-existing 已开启，直接处理当前最后一条")
            elif cur and cur.strip() != prev:
                log(f"[init] 发现未回复消息：{cur!r}（上次回复：{prev!r}）")
                last_seen = prev or None
            else:
                last_seen = cur
                log(f"[init] 当前最后一条对方消息：{last_seen!r}，等待新消息")
                time.sleep(args.interval)
                continue

        if should_reply:
            text = last_in[1]
            dup = last_seen is not None and difflib.SequenceMatcher(None, text, last_seen).ratio() > 0.75
            cooled = (time.time() - last_reply_time) >= args.cooldown
            if dup:
                pass
            elif not cooled:
                log(f"[cool] 冷却中，跳过：{text!r}")
            else:
                log(f"[new] 对方：{text!r}")
                try:
                    reply = get_agent().chat_reply(build_history(incoming, outgoing), chat=title)
                except Exception as e:
                    log(f"[error] 生成回复失败：{e}")
                    time.sleep(args.interval)
                    continue
                log(f"[reply] {reply!r}")
                if not reply:
                    log("[skip] 空回复，不发送")
                elif args.live:
                    try:
                        send_reply(hwnd, reply)
                        log("[sent] 已发送")
                        last_reply_time = time.time()
                        save_state({"last_replied": text, "ts": time.time()})
                    except Exception as e:
                        log(f"[error] 发送失败：{e}")
                else:
                    log("[dry-run] 未发送")
                last_seen = text

        time.sleep(args.interval)


def user32_is_window(hwnd: int) -> bool:
    import ctypes

    return bool(ctypes.windll.user32.IsWindow(hwnd))


if __name__ == "__main__":
    raise SystemExit(main())
