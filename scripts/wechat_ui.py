"""微信助手可视化控制台（规则 8）。

功能：
- 编辑并保存：白名单（单聊/群聊统一管理）、我的群昵称；
- 启动 / 停止多会话自动回复；
- 干跑一次（只扫描不回复）；
- 实时日志。

运行：
    python scripts/wechat_ui.py
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parent.parent
BOT = ROOT / "scripts" / "wechat_multi.py"
LOG = ROOT / "logs" / "wechat_ui.log"
BOT_CONFIG = ROOT / "bot_config.json"
RULES_CONFIG = ROOT / "wechat_rules.json"
PAUSE = ROOT / "PAUSE"
PAUSE_COLLECT = ROOT / "PAUSE_COLLECT"
STOP_COLLECT = ROOT / "STOP_COLLECT"
COLLECT = ROOT / "scripts" / "wechat_collect.py"
COLLECT_HISTORY = ROOT / "scripts" / "wechat_collect_history.py"
INGEST = ROOT / "scripts" / "ingest.py"


def load_json(p: Path, default):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def save_json(p: Path, d) -> None:
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.log_pos = LOG.stat().st_size if LOG.exists() else 0
        self.q: queue.Queue = queue.Queue()
        self.cq: queue.Queue = queue.Queue()

        root.title("个人分身 · 微信助手控制台")
        root.geometry("1040x780")

        self._build()
        self._load()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 打开控制台即暂停任务，避免抢鼠标
        PAUSE.write_text("pause", encoding="utf-8")
        self.root.after(500, self._poll_log)
        self.root.after(150, self._poll_collect)

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        left = ttk.Frame(self.root, padding=10, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(self.root, padding=10)
        right.pack(side="right", fill="both", expand=True)

        # 左：设置（单聊/群聊统一白名单）
        self.lb_multi = self._list_editor(left, "白名单（单聊/群聊统一管理）", 0)
        self.lb_names = self._list_editor(left, "我的群昵称（识别 @我）", 1)

        ttk.Button(left, text="保存设置", command=self._save).grid(
            row=2, column=0, sticky="ew", pady=10
        )

        # 右：状态与控制
        top = ttk.Frame(right)
        top.pack(fill="x")
        self.status_var = tk.StringVar(value="已停止")
        ttk.Label(top, text="状态：", font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.status_var, foreground="#2e7d32").pack(side="left")

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="开始", command=self._start).pack(side="left", padx=2)
        ttk.Button(btns, text="暂停", command=self._pause).pack(side="left", padx=2)
        ttk.Button(btns, text="停止", command=self._stop).pack(side="left", padx=2)
        ttk.Button(btns, text="干跑一次", command=self._dry).pack(side="left", padx=2)
        ttk.Button(btns, text="清空日志", command=self._clear_log).pack(side="left", padx=2)

        col = ttk.LabelFrame(right, text="历史采集（采集期间机器人自动暂停）")
        col.pack(fill="x", pady=4)
        row1 = ttk.Frame(col)
        row1.pack(fill="x")
        ttk.Label(row1, text="来源").pack(side="left")
        self.var_source = tk.StringVar(value="历史记录")
        ttk.Combobox(
            row1,
            textvariable=self.var_source,
            values=["历史记录", "聊天框"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=2)
        ttk.Label(row1, text="屏数").pack(side="left")
        self.var_screens = tk.StringVar(value="10")
        ttk.Entry(row1, textvariable=self.var_screens, width=4).pack(side="left", padx=2)
        ttk.Label(row1, text="步进").pack(side="left")
        self.var_step = tk.StringVar(value="1")
        ttk.Entry(row1, textvariable=self.var_step, width=3).pack(side="left", padx=2)
        ttk.Label(row1, text="采集会话").pack(side="left")
        self.var_chat = tk.StringVar()
        self.cmb_chat = ttk.Combobox(row1, textvariable=self.var_chat, width=10)
        self.cmb_chat.pack(side="left", padx=2)
        self.var_ingest = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="采集后入库", variable=self.var_ingest).pack(side="left", padx=4)

        row2 = ttk.Frame(col)
        row2.pack(fill="x", pady=2)
        ttk.Button(row2, text="开始采集", command=self._collect).pack(side="left", padx=2)
        ttk.Button(row2, text="暂停采集", command=self._pause_collect).pack(side="left", padx=2)
        ttk.Button(row2, text="停止采集", command=self._stop_collect).pack(side="left", padx=2)

        ttk.Label(right, text="运行日志").pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(right, height=28, state="normal")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _list_editor(self, parent, title: str, row: int):
        lf = ttk.LabelFrame(parent, text=title)
        lf.grid(row=row, column=0, sticky="ew", pady=4)
        lb = tk.Listbox(lf, height=3, exportselection=False)
        lb.pack(fill="both", expand=True)
        e = ttk.Entry(lf)
        e.pack(fill="x", pady=(4, 2))
        bf = ttk.Frame(lf)
        bf.pack(fill="x")
        ttk.Button(bf, text="添加", width=6, command=lambda: self._add(lb, e)).pack(side="left")
        ttk.Button(bf, text="删除", width=6, command=lambda: self._remove(lb)).pack(side="left")
        return lb

    # ------------------------------------------------------------------ #
    def _add(self, lb: tk.Listbox, entry: ttk.Entry) -> None:
        v = entry.get().strip()
        if v and v not in lb.get(0, "end"):
            lb.insert("end", v)
            entry.delete(0, "end")

    def _remove(self, lb: tk.Listbox) -> None:
        for i in reversed(lb.curselection()):
            lb.delete(i)

    def _load(self) -> None:
        bc = load_json(BOT_CONFIG, {})
        rules = load_json(RULES_CONFIG, {})
        for lb, key, src in [
            (self.lb_multi, "multi_chats", bc),
            (self.lb_names, "my_names", rules),
        ]:
            lb.delete(0, "end")
            for v in src.get(key, []) or []:
                lb.insert("end", v)
        self.cmb_chat["values"] = list(self.lb_multi.get(0, "end"))

    def _save(self) -> None:
        bc = load_json(BOT_CONFIG, {})
        bc["multi_chats"] = list(self.lb_multi.get(0, "end"))
        save_json(BOT_CONFIG, bc)
        self.cmb_chat["values"] = list(self.lb_multi.get(0, "end"))

        rules = load_json(RULES_CONFIG, {})
        rules["my_names"] = list(self.lb_names.get(0, "end"))
        rules.pop("groups", None)
        rules.pop("thresholds", None)
        rules.pop("interests", None)
        save_json(RULES_CONFIG, rules)
        messagebox.showinfo("已保存", "设置已写入 bot_config.json / wechat_rules.json")

    # ------------------------------------------------------------------ #
    def _start(self) -> None:
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("提示", "机器人已在运行")
            return
        # 开始前解除暂停
        if PAUSE.exists():
            try:
                PAUSE.unlink()
            except OSError:
                pass
        self._save()
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self._log("=== 启动 ===\n")
        self.proc = subprocess.Popen(
            [sys.executable, str(BOT), "--live"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self.status_var.set("运行中")
        threading.Thread(
            target=self._read_stdout, args=(self.proc,), daemon=True
        ).start()
        self.root.after(150, self._poll_queue)

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stdout:
                for line in proc.stdout:
                    self.q.put(line)
        finally:
            self.q.put(None)

    def _poll_queue(self) -> None:
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.status_var.set("已停止")
                break
            self._log(line)
        self.root.after(150, self._poll_queue)

    def _collect(self) -> None:
        for f in (PAUSE_COLLECT, STOP_COLLECT):
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass
        PAUSE.write_text("pause", encoding="utf-8")
        self._log("=== 开始采集历史（机器人已暂停）===\n")
        threading.Thread(target=self._collect_worker, daemon=True).start()

    def _collect_worker(self) -> None:
        screens = self.var_screens.get().strip() or "10"
        step_raw = self.var_step.get().strip() or "1"
        try:
            step_i = int(step_raw)
        except ValueError:
            step_i = 1
        source = self.var_source.get() if hasattr(self, "var_source") else "聊天框"
        chat = self.var_chat.get().strip()

        if source == "历史记录":
            step = str(step_i if step_i >= 4 else 18)
            cmd = [
                sys.executable,
                str(COLLECT_HISTORY),
                "--screens",
                screens,
                "--step",
                step,
                "--auto-open",
            ]
            if chat:
                cmd += ["--chat", chat]
            self.cq.put("提示：需要微信停在目标会话或填写采集会话，脚本会自动打开聊天记录窗口。\n")
        else:
            step = str(max(1, min(3, step_i)))
            cmd = [sys.executable, str(COLLECT), "--screens", screens, "--step", step]
            if chat:
                cmd += ["--chat", chat]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            if p.stdout:
                for line in p.stdout:
                    self.cq.put(line)
            p.wait()
            if self.var_ingest.get():
                self.cq.put("=== 开始入库 ===\n")
                p2 = subprocess.Popen(
                    [sys.executable, str(INGEST)],
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                if p2.stdout:
                    for line in p2.stdout:
                        self.cq.put(line)
                p2.wait()
        except Exception as e:
            self.cq.put(f"[error] 采集失败：{e}\n")
        finally:
            self.cq.put(None)

    def _poll_collect(self) -> None:
        while True:
            try:
                line = self.cq.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self._log("=== 采集结束 ===\n")
                break
            self._log(line)
        self.root.after(150, self._poll_collect)

    def _pause_collect(self) -> None:
        PAUSE_COLLECT.write_text("pause", encoding="utf-8")
        self._log("=== 已暂停采集 ===\n")

    def _stop_collect(self) -> None:
        STOP_COLLECT.write_text("stop", encoding="utf-8")
        self._log("=== 已请求停止采集 ===\n")

    def _pause(self) -> None:
        PAUSE.write_text("pause", encoding="utf-8")
        self.status_var.set("已暂停")
        self._log("=== 已暂停 ===\n")

    def _on_close(self) -> None:
        # 关闭控制台时保持暂停，避免机器人在无人观察时乱点
        try:
            PAUSE.write_text("pause", encoding="utf-8")
        except OSError:
            pass
        self.root.destroy()

    def _stop(self) -> None:
        (ROOT / "STOP").write_text("stop", encoding="utf-8")
        self._log("=== 已请求停止 ===\n")
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
        self.status_var.set("已停止")

    def _dry(self) -> None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self._log("=== 干跑一次 ===\n")
        p = subprocess.run(
            [sys.executable, str(BOT), "--once"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self._log(p.stdout or "")
        if p.stderr:
            self._log(p.stderr)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(text)

    def _poll_log(self) -> None:
        if self.proc and self.proc.poll() is not None and self.status_var.get() == "运行中":
            self.status_var.set("已停止")
        self.root.after(500, self._poll_log)


def main() -> int:
    try:
        root = tk.Tk()
        App(root)
        root.mainloop()
        return 0
    except Exception:
        import traceback

        LOG.parent.mkdir(parents=True, exist_ok=True)
        (LOG.parent / "ui_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
