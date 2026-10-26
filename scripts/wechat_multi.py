"""多会话自动回复（规则 5/6/7）：扫未读红点 → 进入会话 → 判断 → 回复 → 再扫。

安全设计：
- 只有 bot_config.json 里 multi_chats 列出的会话才自动回复；
- 其他会话只“点开已读”，不回复；
- 白名单内的单聊/群聊统一处理，不再做 A/B/C/D 分类；
- 每个会话记录“上次回过的消息”，避免重复。

用法：
    python scripts/wechat_multi.py            # 干跑：只扫描和打印，不回复
    python scripts/wechat_multi.py --live     # 真正回复
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.agent import CloneAgent  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.wechat_store import append_messages, load_messages  # noqa: E402
from src.wechat_unread import detect_unread  # noqa: E402
from src.wechat_win import capture, find_wechat_hwnd, set_dpi_aware, window_rect  # noqa: E402
from src.wechat_layout import detect_chat_x0, detect_chat_x0_ex  # noqa: E402
from src.wechat_vlm import read_row_title  # noqa: E402
import wechat_screener as ws  # noqa: E402


def load_json(p: Path, default):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def save_json(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def history_text(history: list[dict]) -> str:
    return "\n".join(
        f"{'我' if m.get('role') == 'me' else '对方'}：{m.get('text', '')}"
        for m in history[-8:]
    )


TIME_RE = re.compile(
    r"^(\d{1,2}[:：]\d{2}|昨天|前天|星期[一二三四五六日天]"
    r"|\d{1,2}/\d{1,2}|\d{2}月\d{1,2}日|\d{4}/\d{1,2}/\d{1,2})$"
)


def list_titles(engine, img, whitelist: set[str] | None = None) -> list[tuple[str, int]]:
    """OCR 会话列表，返回 [(标题文本, y)]。"""
    w, h = img.size
    x0, found = detect_chat_x0_ex(img)
    # 有聊天区：只裁左侧列表；列表单独显示（没开聊天）才用整窗宽
    right = (x0 - 5) if found else (w - 20)
    crop = img.crop((110, 80, max(300, right), max(120, h - 100)))
    result = ws.ocr_image(engine, crop)
    rows: list[tuple[str, int]] = []
    for item in result or []:
        box, text, score = item[0], str(item[1]), float(item[2])
        if score < 0.45:
            continue
        t = text.strip()
        if not t:
            continue
        # 过滤时间/日期，避免把“19:25”当成会话名
        if TIME_RE.match(t.replace(" ", "")):
            continue
        # 过滤纯数字/符号；但 emoji 群名（如 🐦🐦）和白名单标题要保留
        if re.fullmatch(r"[\d\W_]+", t):
            in_whitelist = bool(whitelist) and any(
                w and (w in t or t in w) for w in whitelist
            )
            is_emoji = bool(re.search(r"[\U0001F000-\U0001FAFF]", t)) and len(t) <= 6
            if not in_whitelist and not is_emoji:
                continue
        # 过滤消息预览（“发送者：内容”、[图片]、[小程序]、撤回提示等）
        if re.search(r"[：:]", t) or t.startswith("[") or "撤回" in t:
            continue
        ys = [p[1] for p in box]
        rows.append((t, int(min(ys)) + 80))

    # 同一行会有“标题 + 预览”两段文字：按 y 聚类，每行只保留最上面的标题
    rows.sort(key=lambda r: r[1])
    deduped: list[tuple[str, int]] = []
    for t, y in rows:
        if deduped and y - deduped[-1][1] < 35:
            continue
        deduped.append((t, y))
    return deduped


def match_whitelist(title: str, whitelist: set[str]) -> str | None:
    if not title:
        return None
    # 去掉标题末尾的“(18)”这类群人数
    base = re.sub(r"[（(]\d+[)）]\s*$", "", title).strip()
    for w in whitelist:
        if w and (w in title or title in w or w in base or base in w):
            return w
    return None


def associate(dot: dict, rows: list[tuple[str, int]], tol: int = 50) -> str:
    """把红点按 y 坐标关联到最近的会话标题。"""
    if not rows:
        return ""
    best = min(rows, key=lambda r: abs(r[1] - dot["y"]))
    return best[0] if abs(best[1] - dot["y"]) <= tol else ""


def load_templates() -> list[tuple[str, "np.ndarray"]]:
    """加载 data/avatars/*.png 作为会话头像模板，文件名即会话名。"""
    templates: list[tuple[str, np.ndarray]] = []
    d = ROOT / "data" / "avatars"
    if d.exists():
        for p in sorted(d.glob("*.png")):
            try:
                img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is not None:
                templates.append((p.stem, img))
    return templates


def match_templates(img, templates, threshold: float = 0.8) -> list[dict]:
    """在会话列表的头像列里找模板，返回匹配到的位置。"""
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    y0, y1 = 80, max(120, h - 100)
    x0, x1 = 55, 112
    roi = arr[y0:y1, x0:x1]
    out: list[dict] = []
    for name, tpl in templates:
        th, tw = tpl.shape[:2]
        if th > roi.shape[0] or tw > roi.shape[1]:
            continue
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv >= threshold:
            out.append(
                {
                    "name": name,
                    "x": maxloc[0] + x0,
                    "y": maxloc[1] + y0,
                    "w": tw,
                    "h": th,
                    "score": float(maxv),
                }
            )
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="真正回复")
    ap.add_argument(
        "--review",
        action="store_true",
        help="复核模式：只生成并显示回复，不发送",
    )
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--once", action="store_true", help="只扫一轮")
    args = ap.parse_args()

    set_dpi_aware()
    pyautogui.FAILSAFE = True
    hwnd = find_wechat_hwnd()
    if not hwnd:
        print("找不到微信窗口")
        return 1

    bot_cfg = load_json(ROOT / "bot_config.json", {})
    review = bool(bot_cfg.get("review_mode", False)) or bool(args.review)
    vlm_model = str(bot_cfg.get("vlm_model") or "")
    rules = load_json(ROOT / "wechat_rules.json", {})
    contacts = load_json(ROOT / "contacts.json", {})
    state_path = ROOT / "data" / "bot_state.json"
    state = load_json(state_path, {})
    chats = state.setdefault("chats", {})

    whitelist = set(bot_cfg.get("multi_chats", []) or [])
    my_names = rules.get("my_names", [])

    agent = CloneAgent(load_config(), ROOT)
    try:
        agent.llm.chat_native([{"role": "user", "content": "ping"}], temperature=0.0)
        print("[warmup] 回复模型已加载")
    except Exception as e:
        print(f"[warmup] 失败：{e}")
    engine = RapidOCR()
    templates = load_templates()
    print(f"[template] 已加载 {len(templates)} 个头像模板")

    def process(title: str) -> str:
        """返回 replied / no_reply / failed；failed 时不要收起聊天框。"""
        got = ws.analyze_once(engine, hwnd)
        last_in = None
        last_out = None
        incoming: list = []
        outgoing: list = []

        # 刚点开时界面可能还没渲染完：重试 + 滚到底
        for attempt in range(4):
            if got is not None:
                last_in, last_out, incoming, outgoing, _t = got
                if last_in is not None:
                    break
            time.sleep(1.2)
            if attempt >= 1:
                # 只滚动聊天消息区；不能用 End 键，否则可能把左侧会话列表滚到底
                left, top, w, h = window_rect(hwnd)
                pyautogui.moveTo(left + int(w * 0.72), top + int(h * 0.5))
                pyautogui.scroll(-1200)
            got = ws.analyze_once(engine, hwnd)

        if got is None or last_in is None:
            print("  [warn] 未能读取到会话消息，保留聊天框")
            return "failed"

        # 最后一条是我方 -> 无需回
        if last_out is not None and last_in[0] <= last_out[0]:
            print("  [skip] 最后一条是我方，无需回复")
            return "no_reply"
        text = last_in[1]
        prev = chats.get(title, {})
        # 同一条消息 120 秒内只回一次；避免重复表情（如 😀😀😀）被永久去重
        if prev.get("last_replied") == text and time.time() - prev.get("ts", 0) < 120:
            print("  [skip] 刚回复过同一条消息（120 秒内去重）")
            return "no_reply"

        # 3 秒防抖：继续等新消息，直到一轮没有新消息再一起回
        history = ws.build_history(incoming, outgoing)
        collected = [text]
        for _ in range(10):
            time.sleep(3)
            nxt = ws.analyze_once(engine, hwnd)
            if nxt is None:
                break
            n_last_in, n_last_out, n_incoming, n_outgoing, n_title = nxt
            if not n_last_in or not n_title:
                break
            if n_last_out is not None and n_last_in[0] <= n_last_out[0]:
                break
            if n_last_in[1] == collected[-1]:
                break  # 这一轮没有新消息，安静了
            collected.append(n_last_in[1])
            history = ws.build_history(n_incoming, n_outgoing)
            print(f"  [collect] 又收到：{n_last_in[1]!r}")

        # 上一条已存消息距现在多久（用于判断是否换了话题）
        gap_minutes = None
        try:
            stored = load_messages(title)
            if stored:
                last_ts = time.mktime(
                    time.strptime(str(stored[-1].get("ts"))[:16], "%Y-%m-%d %H:%M")
                )
                gap_minutes = (time.time() - last_ts) / 60
        except Exception:
            gap_minutes = None

        # 实时入库：时间用当前时间（截图里没有稳定时间），格式与批量采集一致
        try:
            now_ts = time.strftime("%Y-%m-%d %H:%M")
            live_msgs = [
                {
                    "ts": now_ts,
                    "role": m.get("role"),
                    "sender": m.get("sender"),
                    "text": m.get("text"),
                    "source": "live",
                }
                for m in history
                if str(m.get("text") or "").strip()
            ]
            if live_msgs:
                added = append_messages(title, live_msgs)
                if added:
                    print(f"  [store] +{added} 条")
        except Exception as e:
            print(f"  [warn] 实时入库失败：{e}")

        if not args.live:
            print(f"  [dry] 会回复 {title!r}：{collected!r}")
            return "no_reply"

        rounds = 0
        last_text = collected[-1]
        while rounds < 3:
            reply = agent.chat_reply(history, chat=title, gap_minutes=gap_minutes)
            gap_minutes = None
            if not reply:
                print(f"  [skip] 模型返回空回复：{agent.last_skip_reason or '未知原因'}")
                break
            if review:
                print(f"  [review] {title!r} <- {reply!r}")
                try:
                    logdir = ROOT / "logs"
                    logdir.mkdir(parents=True, exist_ok=True)
                    with open(logdir / "review.log", "a", encoding="utf-8") as f:
                        f.write(f"{time.strftime('%F %T')}\t{title}\t{reply}\n")
                except Exception:
                    pass
                chats[title] = {"last_replied": last_text, "ts": time.time()}
                save_json(state_path, state)
                return "review"
            ws.send_reply(hwnd, reply)
            rounds += 1
            print(f"  [sent] {title!r} <- {reply!r}")
            chats[title] = {"last_replied": last_text, "ts": time.time()}
            save_json(state_path, state)

            # 跟聊：等 3 秒看有没有新消息；连续 2 次安静就收起
            new_ctx = None
            for _ in range(2):
                time.sleep(3)
                nxt = ws.analyze_once(engine, hwnd)
                if nxt is None:
                    continue
                n_last_in, n_last_out, n_in, n_out, _t = nxt
                if not n_last_in or n_last_in[1] == last_text:
                    continue
                if n_last_out is not None and n_last_in[0] <= n_last_out[0]:
                    continue
                last_text = n_last_in[1]
                new_ctx = (n_in, n_out, n_last_in)
                break
            if new_ctx is None:
                break
            n_in, n_out, n_last_in = new_ctx
            history = ws.build_history(n_in, n_out)
            collected = [n_last_in[1]]
            print(f"  [follow] 新消息：{n_last_in[1]!r}")

        return "replied" if rounds else "no_reply"

    paused_logged = False
    last_dot = {"x": -1, "y": -1, "ts": 0.0}
    skip_log: dict[str, float] = {}
    while True:
        if (ROOT / "STOP").exists():
            try:
                (ROOT / "STOP").unlink()
            except OSError:
                pass
            print("收到 STOP，退出")
            return 0
        if (ROOT / "PAUSE").exists():
            if not paused_logged:
                print("[paused] 已暂停（控制台打开/手动暂停），不扫描不点击")
                paused_logged = True
            time.sleep(0.5)
            continue
        paused_logged = False
        left, top, w, h = window_rect(hwnd)
        img = capture(hwnd)
        if img is None:
            time.sleep(args.interval)
            continue
        dots = detect_unread(img)
        if not dots:
            print("[scan] 没有未读")
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        rows = list_titles(engine, img, whitelist)
        tmatches = match_templates(img, templates)
        for d in dots:
            title = associate(d, rows)
            target = match_whitelist(title, whitelist)
            # 只有 OCR 完全没读出来（空/单字）时才让视觉模型兜底；
            # 能读出名字但不是白名单的，直接静默跳过
            if not target and vlm_model and (
                len(title.strip()) <= 1 or "：" in title or ":" in title
            ):
                try:
                    vtitle = read_row_title(
                        agent.llm, img, d["y"], detect_chat_x0(img), vlm_model
                    )
                except Exception:
                    vtitle = ""
                cand = match_whitelist(vtitle or "", whitelist)
                if cand:
                    print(f"[vlm] 红点行识别为 {vtitle!r}（OCR: {title!r}）")
                    target = cand
            if not target and tmatches:
                for m in tmatches:
                    if (
                        m["x"] + m["w"] - 8 <= d["x"] <= m["x"] + m["w"] + 14
                        and m["y"] - 8 <= d["y"] <= m["y"] + 20
                    ):
                        target = match_whitelist(m["name"], whitelist)
                        if target:
                            print(f"[template] 红点匹配头像 {m['name']!r}（相似度 {m['score']:.2f}）")
                        break
            if not target:
                continue  # 非白名单：静默跳过，不打印
            if not args.live:
                print(f"[dry] 白名单未读 {target!r} @({d['x']},{d['y']})")
                continue
            if (
                d["x"] == last_dot["x"]
                and d["y"] == last_dot["y"]
                and time.time() - last_dot["ts"] < 10
            ):
                print("[skip] 同一红点刚点过，避免反复点进")
                continue
            try:
                current = ws.read_title(engine, img)
            except Exception:
                current = ""
            if current and target in current:
                print(f"[keep] 当前已是 {target!r}，不重复点击")
            else:
                print(f"[click] {target!r} @({d['x']},{d['y']})")
                pyautogui.click(left + d["x"], top + d["y"])
            last_dot = {"x": d["x"], "y": d["y"], "ts": time.time()}
            time.sleep(1.8)
            status = process(target)
            if status == "failed":
                print("[warn] 未读取到消息，不收起聊天框（避免红点丢失）")
                time.sleep(2)
                continue
            # 收起聊天框：再点一次同一会话
            # 否则停留在该会话里，它收到的新消息不会在列表出现红点
            pyautogui.click(left + d["x"], top + d["y"])
            time.sleep(0.8)
            time.sleep(3)  # 规则 6：回完等 3 秒再看

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
