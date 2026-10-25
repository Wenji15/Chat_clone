"""Qwen2.5-VL-7B LoRA 微调：用聊天语料学“说话风格”，视觉能力保留。

与 train_lora.py 的区别：
- 基座换成 Qwen2_5_VLForConditionalGeneration（带视觉编码器）
- 文本对话数据（无图片）同样可训，只影响语言部分
- LoRA 只挂在语言层的 q/k/v/o/gate/up/down 上，视觉塔冻结

用法：
    python training/train_vlm_lora.py --max-steps 2          # 冒烟测试
    python training/train_vlm_lora.py --epochs 1 --lr 1e-4   # 正式训练
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="training/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data", default="data/finetune/sft_all_train.jsonl")
    ap.add_argument("--val", default="data/finetune/sft_all_val.jsonl")
    ap.add_argument("--output", default="training/vlm-output")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=-1, help="调试用：只训几步")
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": args.data, "val": args.val})
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_length=args.max_length,
        assistant_only_loss=True,
        max_steps=args.max_steps,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    print(f"训练完成，VLM LoRA 适配器已保存到 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
