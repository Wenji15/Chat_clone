"""把 VLM 的 LoRA 适配器合并回 Qwen2.5-VL 基座。

用法：
    python training/merge_vlm_lora.py \
        --model training/models/Qwen2.5-VL-7B-Instruct \
        --adapter training/vlm-output \
        --output training/vlm-merged
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="training/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="training/vlm-output")
    ap.add_argument("--output", default="training/vlm-merged")
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.output)

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    proc.save_pretrained(args.output)
    print(f"VLM 合并完成 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
