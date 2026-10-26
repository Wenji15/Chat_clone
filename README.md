# 我的个人分身（AI Clone）

> ⚠️ **免责声明**
> 本项目通过非官方接口/模拟操作控制微信，**违反微信用户协议，存在封号风险**。
> **禁止在日常微信号上测试**，请使用小号/测试号，仅作编程技术学习参考，使用后果自负。

一个**本地运行、隐私可控**的个人 AI 分身骨架，覆盖你想要的三大能力：

| 能力 | 实现方式 | 对应文件 |
|------|---------|---------|
| **口吻像你** | 人设 Prompt（persona.md） | `persona.md`、`src/agent.py` |
| **知道你所知道的** | 个人知识库 + RAG 检索 | `src/memory.py`、`scripts/ingest.py` |
| **替你干活** | 工具调用（Function Calling） | `src/tools.py` |

设计原则：**零第三方依赖**——只用 Python 标准库，推理和向量化全部交给本地 Ollama，你的现有 Python 直接就能跑，不用装任何包。

---

## 目录结构

```
AI_clone/
├── config.json            # 配置：模型名、路径、参数
├── persona.md             # ★ 人格设定（改成你自己的口吻）
├── requirements.txt       # 依赖说明（零依赖）
├── data/
│   ├── raw/               # ★ 放你的个人资料（.txt/.md/.csv/.json）
│   ├── knowledge_store.json  # 生成：向量知识库
│   └── memory.jsonl          # 生成：长期记忆
├── src/
│   ├── cli.py             # 命令行聊天入口
│   ├── web.py             # 内置 Web 聊天界面入口
│   ├── agent.py           # Agent 主循环（人设+RAG+工具）
│   ├── ollama_client.py   # Ollama 客户端（对话/嵌入）
│   ├── memory.py          # 向量库 + 长期记忆
│   ├── tools.py           # 工具集（时间/读写文件/记住）
│   └── config_loader.py   # 配置加载
├── scripts/ingest.py      # 资料切片+嵌入+入库
└── web/index.html         # Web 聊天前端
```

---

## 快速开始（5 步）

### 1. 安装 Ollama

去 <https://ollama.com> 下载 Windows 安装包，装好后保持它在后台运行（托盘有 🦙 图标）。

### 2. 拉取模型

本项目默认使用以下两个模型（你本地已装好，可直接跳过）：

```bash
ollama pull qwen3.5:9b   # 对话模型（6.6GB，8G 显存友好）
ollama pull bge-m3       # 嵌入模型（中文检索强，1.2GB）
```

> 想换模型：改 `config.json` 里的 `chat_model` / `embed_model` 即可。
> 你本地可选的对话模型还有 `qwen3.5:9b-q8_0`（精度更高但更吃显存）、`gemma4:12b`、`qwen9b-long` 等。

### 3. 编辑人设

打开 `persona.md`，把「称呼 / 说话风格 / 口头禅 / 语气」改成你自己的，删掉示例里的下划线占位符。

### 4. 放进你的资料

把你的笔记、博客、聊天记录整理成 `.txt` / `.md` / `.csv` / `.json`，丢进 `data/raw/`（已有一个示例 `example.md`，可删可改）。

然后导入到知识库（在项目根目录执行）：

```bash
python scripts/ingest.py
```

### 5. 启动分身

命令行聊天：

```bash
python -m src.cli
```

或内置 Web 界面（更推荐，体验更好）：

```bash
python -m src.web
```

浏览器打开 <http://127.0.0.1:8000> 即可聊天。

### 6. 自检（可选）

如果启动报错，先跑一遍自检，它会告诉你 Ollama 连接、模型、知识库哪一环出了问题：

```bash
python scripts/check.py
```

---

## 分身现在会做什么

- **聊天的口吻像你**：system prompt 里加载了 `persona.md` 的人设。
- **回答基于你的知识**：你的问题会先在个人知识库里检索相关内容，再交给模型回答，模型被要求忠于检索内容、不编造。
- **能执行任务**：它会按需调用工具——获取当前时间、读写项目目录内的文件、列出目录、把重要信息写进长期记忆。
- **有长期记忆**：每轮对话自动沉淀到 `data/memory.jsonl`，后续对话会把最近记忆注入上下文；它也可以主动调用 `remember` 工具记住关键信息。

---

## 常见问题

**Q：提示「无法连接 Ollama」？**
确认 Ollama 在运行，且 `config.json` 里 `base_url` 是 `http://127.0.0.1:11434`。

**Q：导入时报「嵌入接口响应异常」？**
多半是没拉嵌入模型：执行 `ollama pull bge-m3`。

**Q：回复很慢？**
本地模型速度取决于显卡。7B 量化模型在 8G 显存上通常几秒到十几秒一答；CPU 模式会明显更慢。

**Q：换了模型要重新导入吗？**
知识库的向量与对话模型无关，只与嵌入模型有关，所以换对话模型不需要重新 `ingest`。

---

## 进阶路线（想做得更像「完整克隆」）

1. **换更大的模型**：显存 16G+ 可换 `gemma4:12b`（你本地已有）或更大的 Qwen3.5 版本，口吻和逻辑会更稳；8G 显存先用 `qwen3.5:9b`。
2. **喂更多个人语料**：聊天记录、朋友圈、微博、邮件——数据量越大，"像不像"的上限越高。
3. **LoRA 微调**：当 Prompt + RAG 不足以抓住你的独特口吻时，用你的语料对 7B/14B 做 LoRA 微调（需要更多数据和 GPU，建议等骨架跑顺后再做）。
4. **加更多工具**：在 `src/tools.py` 里按同样格式加新工具（发邮件、查天气、跑脚本等），注意高权限操作要加白名单或二次确认。
5. **换更专业的向量库**：数据量上到上万条后，把 `memory.py` 里的纯 Python 检索换成 Chroma / Qdrant / FAISS。

---

## RAG 与微调

> 完整流程、原理、参数与排错见 **[docs/pipeline.md](docs/pipeline.md)**。

### RAG（检索增强，不改模型）

```powershell
# 1. 资料放入 data/raw/（微信聊天记录会自动导出到 data/raw/wechat/）
# 2. 切块 + 嵌入
python scripts/ingest.py
# 3. 提问（检索 + 本地模型回答）
python scripts/query.py "闫总是谁"
```

- 向量模型：`bge-m3`（`config.json` 的 `embed_model`）
- 知识库：`data/knowledge_store.json`
- 检索结果会打印命中的来源片段

### 微调（LoRA，改变说话方式）

完整步骤见 [training/README.md](training/README.md)：

```powershell
python scripts/export_sft.py --all          # 导出训练样本
pip install -r training/requirements.txt    # 装训练依赖（torch 等）
python training/train_lora.py --model training\models\Qwen2.5-7B-Instruct --epochs 2 --lr 1.2e-4
python training/merge_lora.py --model training\models\Qwen2.5-7B-Instruct --adapter training\output-all --output training\merged-all
# 转 GGUF 后用 ollama create 导入
```

> 16G 显存用 7B + 4bit QLoRA；8G 显存可换 3B。数据量少时先别微调，RAG 更划算。

## 隐私提示

全部数据（模型、知识库、记忆、对话记录）都在本机，不会上传云端。但 `data/memory.jsonl` 与 `data/knowledge_store.json` 包含你的隐私，已被 `.gitignore` 忽略，**切勿上传到公网仓库**。
