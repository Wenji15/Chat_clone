"""按标准名生成个人档案：data/people/<名>.md

用法：
    python scripts/build_people.py                 # 所有名单内的人
    python scripts/build_people.py --person 田政
    python scripts/build_people.py --min-messages 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402
from src.wechat_store import (  # noqa: E402
    STORE,
    _load_names,
    clean_sender,
    load_messages,
    safe_name,
)

CHUNK = 40


def collect_by_person() -> dict[str, list[dict]]:
    names, alias = _load_names()
    out: dict[str, list[dict]] = {n: [] for n in names}
    for d in sorted(STORE.iterdir()):
        if not d.is_dir():
            continue
        chat = d.name
        for m in load_messages(chat):
            if m.get("role") != "them":
                continue
            name = clean_sender(m.get("sender"))
            if not name:
                continue
            out.setdefault(name, []).append(
                {"chat": chat, "ts": m.get("ts", ""), "text": m.get("text", "")}
            )
    return out


def summarize_chunk(client: OllamaClient, name: str, rows: list[dict]) -> str:
    body = "\n".join(f"[{r['ts']}] {r['text']}" for r in rows)
    prompt = (
        f"下面是我和「{name}」的聊天片段（我是吴恩吉，{name} 是对方）。"
        "请只根据这些内容，提炼关于对方的要点：\n"
        "- 关系/身份\n- 性格\n- 说话风格（句长、口头禅、梗）\n"
        "- 常聊话题\n- 对我的态度\n- 近期状态\n"
        "- 有代表性的原话 1-3 条\n"
        "不确定的写“不确定”，不要编造。用简洁中文条目。\n\n"
        + body
    )
    return client.chat_native(
        [{"role": "user", "content": prompt}], temperature=0.3
    ).strip()


def finalize(client: OllamaClient, name: str, notes: str, total: int, span: str) -> str:
    prompt = (
        f"请把下面关于「{name}」的零散笔记合并成一份档案（Markdown）。"
        "保留有依据的内容，去掉重复；不确定的标注“不确定”。"
        "结构：关系、性格、说话风格、常聊话题、对我的态度、近期状态、口头禅/梗、原话依据。\n\n"
        f"消息数：{total}，时间范围：{span}\n\n{notes}"
    )
    body = client.chat_native(
        [{"role": "user", "content": prompt}], temperature=0.3
    ).strip()
    return f"# {name}\n\n> 消息数：{total}，时间范围：{span}\n\n{body}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="")
    ap.add_argument("--min-messages", type=int, default=30)
    ap.add_argument("--max-messages", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config()
    client = OllamaClient(
        base_url=cfg["ollama"]["base_url"],
        chat_model=cfg["ollama"]["chat_model"],
        embed_model=cfg["ollama"]["embed_model"],
    )
    by_person = collect_by_person()
    targets = [args.person] if args.person else sorted(by_person)

    outdir = ROOT / "data" / "people"
    outdir.mkdir(parents=True, exist_ok=True)

    for name in targets:
        rows = by_person.get(name) or []
        if len(rows) < args.min_messages:
            print(f"[skip] {name}: {len(rows)} 条 < {args.min_messages}")
            continue
        rows.sort(key=lambda r: str(r.get("ts", "")))
        rows = rows[-args.max_messages :]
        span = f"{rows[0]['ts']} ~ {rows[-1]['ts']}" if rows else ""
        notes = []
        for i in range(0, len(rows), CHUNK):
            part = rows[i : i + CHUNK]
            notes.append(summarize_chunk(client, name, part))
            print(f"[{name}] {i + len(part)}/{len(rows)}")
        md = finalize(client, name, "\n\n".join(notes), len(rows), span)
        path = outdir / f"{safe_name(name)}.md"
        path.write_text(md, encoding="utf-8")
        print(f"[ok] {name} -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
