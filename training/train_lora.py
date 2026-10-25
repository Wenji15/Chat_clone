"""LoRA / QLoRA 微调个人分身模型（v2：只学“怎么接话”，带场景）。

关键改动：
- 只对 assistant 回复算 loss（assistant_only_loss），避免模型背上下文
- system 里带场景（群名 + 对话对象），学会“不同场合跟不同人说话”
- 防过拟合：1 epoch、lr 1e-4、LoRA dropout 0.1、weight decay 0.01

前置：
    pip install -r training/requirements.txt

用法：
    python training/train_lora.py --model training/models/Qwen2.5-7B-Instruct --epochs 1
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# 只对 assistant 回复算 loss 的聊天模板（trl 靠 {% generation %} 定位回复段）
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
    ap.add_argument("--model", default="training/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data/finetune/sft_all_train.jsonl")
    ap.add_argument("--val", default="data/finetune/sft_all_val.jsonl")
    ap.add_argument("--output", default="training/output-v3")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=-1, help="调试用：只训几步")
    ap.add_argument("--save-steps", type=int, default=50, help="每多少步存一次检查点")
    ap.add_argument("--no-4bit", action="store_true", help="显存足够时可关闭 4bit")
    args = ap.parse_args()

    ds = load_dataset("json", data_files={"train": args.data, "val": args.val})
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
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
        save_total_limit=4,
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
    print(f"训练完成，LoRA 适配器已保存到 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
