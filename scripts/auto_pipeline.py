"""一键流水线：导出 SFT → 重建 RAG →（可选）重训 → 合并 → 转 GGUF → 导入 Ollama。

默认只做“数据 + RAG”更新（几十秒）；加 --train 才跑训练（约 25 分钟）。

用法：
    python scripts/auto_pipeline.py              # 导出 + 重建 RAG
    python scripts/auto_pipeline.py --train      # 全量：训练 + 合并 + 转换 + 导入
    python scripts/auto_pipeline.py --train --epochs 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
MODEL_NAME = "clone-richer-v3"
ADAPTER = "training/output-v3"
MERGED = "training/merged-v3"
GGUF = "training/merged-v3-q8.gguf"


def run(cmd: list[str], desc: str) -> None:
    print(f"\n=== {desc} ===")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"❌ 失败：{desc}（exit {result.returncode}）")
        sys.exit(result.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="包含重训与模型更新")
    ap.add_argument("--model", default=str(ROOT / "training" / "models" / "Qwen2.5-7B-Instruct"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--skip-ingest", action="store_true", help="跳过重建 RAG")
    args = ap.parse_args()

    run([PY, "scripts/export_sft.py", "--all"], "1/6 导出 SFT 数据")

    if not args.skip_ingest:
        run([PY, "scripts/ingest.py"], "2/6 重建 RAG 向量库")

    if not args.train:
        print("\n✅ 数据与 RAG 已更新（未重训）。加 --train 可重训模型。")
        return 0

    # 释放显存（Ollama 常驻会占 8G+）
    subprocess.run(["ollama", "stop", MODEL_NAME], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    run([
        PY, "training/train_lora.py",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output", ADAPTER,
    ], "3/6 训练 LoRA")
    run([
        PY, "training/merge_lora.py",
        "--model", args.model,
        "--adapter", ADAPTER,
        "--output", MERGED,
    ], "4/6 合并适配器")
    run([
        PY, "training/convert_hf_to_gguf.py",
        MERGED,
        "--outfile", GGUF,
        "--outtype", "q8_0",
    ], "5/6 转 GGUF")
    run(["ollama", "create", MODEL_NAME, "-f", "training/Modelfile"], "6/6 导入 Ollama")

    print(f"\n✅ 全流程完成：{MODEL_NAME} 已更新")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
