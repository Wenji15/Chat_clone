"""内置 Web 聊天界面：python -m src.web

零第三方依赖：用标准库 http.server 提供静态页面 + /api/chat 接口，
浏览器打开 http://127.0.0.1:<web_port> 即可和分身聊天。
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .agent import CloneAgent
from .config_loader import ROOT, load_config

WEB_DIR = ROOT / "web"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


class Handler(BaseHTTPRequestHandler):
    agent: CloneAgent | None = None  # 由 main() 注入

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:
        path = urlparse(self.path).path or "/"
        if path == "/":
            path = "/index.html"
        fp = (WEB_DIR / path.lstrip("/")).resolve()
        if fp.is_file() and str(fp).startswith(str(WEB_DIR.resolve())):
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _MIME.get(fp.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    # ------------------------------------------------------------------ #
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/chat", "/api/chat_reply"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        try:
            if not self.agent:
                reply = "服务未就绪。"
            elif path == "/api/chat":
                reply = self.agent.answer(str(body.get("message", "")).strip())
            else:
                reply = self.agent.chat_reply(list(body.get("messages", [])))
        except Exception as e:
            reply = f"出错：{e}"

        data = json.dumps({"reply": reply}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # 静默访问日志
    def log_message(self, *args) -> None:
        pass


def main() -> int:
    cfg = load_config()
    Handler.agent = CloneAgent(cfg, ROOT)
    port = int(cfg.get("web_port", 8000))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"🌐 个人分身 Web 界面已启动，浏览器打开：http://127.0.0.1:{port}")
    print("   按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
