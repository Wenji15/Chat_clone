"""命令行入口：python -m src.cli"""
from __future__ import annotations

from .agent import CloneAgent
from .config_loader import ROOT, load_config


def main() -> int:
    cfg = load_config()
    agent = CloneAgent(cfg, ROOT)
    print(f"🤖 个人分身已启动（模型：{cfg['ollama']['chat_model']}）")
    print("   输入内容开始对话；输入 /exit 或按 Ctrl+C 退出。")

    while True:
        try:
            user = input("\n你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见 👋")
            return 0

        if not user:
            continue
        if user.lower() in {"/exit", "/quit", "exit", "quit"}:
            print("再见 👋")
            return 0

        try:
            reply = agent.answer(user)
        except Exception as e:
            print(f"出错：{e}")
            continue
        print(f"\n分身 > {reply}")


if __name__ == "__main__":
    raise SystemExit(main())
