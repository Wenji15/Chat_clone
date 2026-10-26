# 本地微调（LoRA / QLoRA）流程

从聊天记录训练一个"像群里某人说话"的本地模型。

## 0. 前置条件

- Windows + NVIDIA 显卡（8G 显存起步）
- 已安装 CUDA 版 PyTorch（`pip install torch --index-url https://download.pytorch.org/whl/cu121`）
- 磁盘预留 20GB（模型权重 + 训练输出）

```powershell
# RTX 50 系列（Blackwell）必须用 cu128 版；国内用阿里云镜像
pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
# 其余依赖走清华 PyPI 镜像
pip install -r training/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# HuggingFace 模型下载镜像
$env:HF_ENDPOINT="https://hf-mirror.com"
```

> RTX 5070 Ti 16G：默认 7B + 4bit QLoRA 就够用。
> 8G 显存：先下载 `Qwen/Qwen2.5-3B-Instruct`，再用 `--model` 指向本地目录。

## 1. 准备数据

```powershell
# 学所有人（每个发送者一条样本，推荐先跑这个）
python scripts/export_sft.py --all

# 或只学"我"
python scripts/export_sft.py
```

输出：

```
data/finetune/sft_all.jsonl        # 全量
data/finetune/sft_all_train.jsonl  # 训练集（90%）
data/finetune/sft_all_val.jsonl    # 验证集（10%）
```

> 当前语料约 2745 条，全量 SFT 2621 条（train 2358 / val 263）；"只学我" 477 条。

## 2. 训练

```powershell
python training/train_lora.py `
  --model training\models\Qwen2.5-7B-Instruct `
  --data data/finetune/sft_all_train.jsonl `
  --val data/finetune/sft_all_val.jsonl `
  --output training\output-all `
  --epochs 2 --lr 1.2e-4
```

v2 起的关键改进：

- **assistant-only loss**：只对"回复"算 loss，不再背上下文（避免吐出一整段聊天记录）
- **场景化 system**：`你正在扮演「X」。场景：微信群「Y」，正在和「Z」等人聊天。`
- **防过拟合**：`lr 1.2e-4`、LoRA dropout 0.1、weight decay 0.01、cosine 调度
- **学会停**：`<|im_end|>` 放在 `{% generation %}` 掩码内

> 如果 loss 后期掉到 1 以下、eval loss 反而上升，就是过拟合，减少 epoch 或加数据。

## 3. 合并适配器

```powershell
python training/merge_lora.py `
  --model training\models\Qwen2.5-7B-Instruct `
  --adapter training/output-all `
  --output training/merged-all
```

## 4. 转成 GGUF 并导入 Ollama

```powershell
# 下载 llama.cpp（只需一次；国内用 ghproxy）
curl.exe -L "https://ghproxy.net/https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip" -o llama.zip
Expand-Archive llama.zip -DestinationPath . -Force

# 转 q8_0 GGUF（转换脚本只支持 f16/bf16/q8_0）
python llama.cpp-master\convert_hf_to_gguf.py training\merged-all --outfile training\merged-all-q8.gguf --outtype q8_0

# 导入 Ollama（clone-all.Modelfile 指向 merged-all-q8.gguf）
ollama create clone-all -f training\clone-all.Modelfile
ollama run clone-all
```

然后在 `config.json` 里把 **`chat_reply_model`** 改成 `clone-all`（回复走微调模型，分类/意图仍用基础模型）。

## 5. 效果验证

```powershell
ollama run clone-richer "在吗，周末踢球不"
ollama run clone-richer "今晚吃啥"
```

看回复是否像目标人物：短句、口头禅、语气。

## 注意事项

1. **数据量**：725 条只够"尝鲜级"微调，想稳定学口吻建议 2000 条以上。
2. **清洗**：OCR 错字、争吵内容会带偏模型，训练前建议手动过一遍。
3. **过拟合**：样本少时 `--epochs` 调小（1–2），否则模型会背原句。
4. **显存不足**：把 `--model` 换更小的（Qwen2.5-1.5B），或降低 `--max-length`。
5. **只学一个人**：`export_sft.py --all` 会给每个发送者生成样本，system 里指定"扮演谁"；也可以按人筛选后单独训练。

## 与 RAG 的关系

- **RAG**：检索你的资料来回答（`scripts/ingest.py` + `scripts/query.py`），不改模型
- **微调**：改变模型的说话方式
- 两者可叠加：先用 RAG 保证事实，再用微调保证口吻
