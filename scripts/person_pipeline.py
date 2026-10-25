"""按人训练：导出 → 训练 LoRA → 合并 → 转 GGUF → 导入 Ollama。

用法：
    python scripts/person_pipeline.py --person 王宇航
    python scripts/person_pipeline.py --all --min-samples 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = sys.executable
BASE_MODEL = ROOT / "training" / "models" / "Qwen2.5-7B-Instruct"
PERSONS_DIR = ROOT / "data" / "finetune" / "persons"
OUT_DIR = ROOT / "training" / "persons"

# Ollama 模型名只支持 ASCII，给常用的人起英文名
MODEL_ALIASES = {
    "王宇航": "wangyuhang",
    "田政": "tianzheng",
    "小江": "xiaojiang",
    "苏博": "subo",
    "盖瑞": "gairy",
    "马灯": "madeng",
    "闫啸SEAN": "sean",
    "Richer": "richer",
}


def model_name(name: str) -> str:
    ascii_name = MODEL_ALIASES.get(name)
    if not ascii_name:
        ascii_name = "p" + str(abs(hash(name)) % 100000)
    return f"clone-{ascii_name}"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} 失败（exit {r.returncode}）")


def train_one(name: str, epochs: float, lr: float) -> None:
    from src.wechat_store import safe_name

    safe = safe_name(name)
    train = PERSONS_DIR / f"{safe}_train.jsonl"
    val = PERSONS_DIR / f"{safe}_val.jsonl"
    if not train.exists():
        print(f"跳过 {name}：没有 {train.name}")
        return

    adapter = OUT_DIR / safe
    merged = OUT_DIR / f"{safe}_merged"
    gguf = OUT_DIR / f"{safe}_q8.gguf"
    modelfile = OUT_DIR / f"{safe}.Modelfile"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(["ollama", "stop", model_name(name)], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    run([PY, "training/train_lora.py",
         "--model", str(BASE_MODEL),
         "--data", str(train), "--val", str(val),
         "--output", str(adapter),
         "--epochs", str(epochs), "--lr", str(lr)], f"训练 {name}")
    run([PY, "training/merge_lora.py",
         "--model", str(BASE_MODEL),
         "--adapter", str(adapter), "--output", str(merged)], f"合并 {name}")
    run([PY, "training/convert_hf_to_gguf.py",
         str(merged), "--outfile", str(gguf), "--outtype", "q8_0"], f"转 GGUF {name}")

    modelfile.write_text(
        f"FROM {gguf.as_posix()}\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        f'SYSTEM "你正在扮演「{name}」，请按他/她平时的说话方式回复。"\n',
        encoding="utf-8",
    )
    run(["ollama", "create", model_name(name), "-f", str(modelfile)], f"导入 {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", action="append", default=[], help="人名，可重复")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    args = ap.parse_args()

    if args.all:
        subprocess.run([PY, "scripts/export_person.py",
                        "--min-samples", str(args.min_samples)], cwd=ROOT, check=True)
        names = [
            p.name[: -len("_train.jsonl")]
            for p in PERSONS_DIR.glob("*_train.jsonl")
        ]
    else:
        names = args.person

    if not names:
        print("请用 --person 指定人名，或 --all 训练所有达标的人")
        return 1

    for name in names:
        try:
            train_one(name, args.epochs, args.lr)
        except RuntimeError as e:
            print(f"❌ {name}: {e}")
            continue

    print("\n✅ 按人训练完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
