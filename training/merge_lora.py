"""把 LoRA 适配器合并回基础模型，得到可直接转换的完整模型。

用法：
    python training/merge_lora.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --adapter training/output \
        --output training/merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="training/output")
    ap.add_argument("--output", default="training/merged")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.output)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(args.output)
    print(f"合并完成 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
