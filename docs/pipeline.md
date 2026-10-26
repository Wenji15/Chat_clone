# 个人分身：完整流程与原理

> ⚠️ **免责声明**
> 本项目通过非官方接口/模拟操作控制微信，**违反微信用户协议，存在封号风险**。
> **禁止在日常微信号上测试**，请使用小号/测试号，仅作编程技术学习参考，使用后果自负。

> 从微信聊天记录 → 语料 → RAG / 微调 → 微信自动回复，全链路说明。

## 1. 当前架构（最新）

### 模型栈

| 用途 | 模型 | 上下文 | 说明 |
|------|------|--------|------|
| 微信回复 / 意图 / 分类 | `qwen3.5:9b-q8_0-ctx8k` | 8k | 基础模型，think 关闭 |
| web/RAG 问答 | 同上 | 8k | think 可开（`thinking_answer`） |
| 回复口吻（可选） | `clone-all`（微调） | — | 训练完由 `chat_reply_model` 指定 |
| 红点行兜底识别 | `qwen2.5vl:3b-ctx` | 4k | 仅 OCR 读不出标题时调用 |
| 历史检索 / RAG | `bge-m3` | — | 向量嵌入 |

### 微信回复链路

```
红点 → 打开会话 → OCR 整屏（最多 40 条）
  ├─ 画像（data/profiles/<会话>.md）：性格/关系/态度/经历
  ├─ 历史检索（bge-m3，top-3，分数 ≥0.45）
  ├─ 意图分析（模型输出 intent / topic / addressed_to_me）
  ├─ 风格示例（该会话“我”的 5 句话）
  ├─ 名词速查（entity_notes.json 优先，Bing 2 秒兜底）
  └─ 生成回复（think=false，最多 2 句）→ 后处理
```

### 关键规则

- 群内**邀约**（约球/约麻将/约饭）**没点名我** → 不回复
- 调侃 → 自嘲或反击，不转移话题
- 不知道的信息 → “再看吧/再说/我问下”，不编造
- 不说脏话；不主动约时间/提议聚聚
- 书面化短语替换（“挺有意思” → “有点东西”）

### 延迟（实测）

| 环节 | 耗时 |
|------|------|
| 分类 | ~0.6s |
| 意图 | ~0.6s |
| 回复 | ~0.6s |
| 冷启动（首次加载） | ~8s |

> 提速关键：`think:false` + `num_ctx 8k` + `keep_alive 30m` + 启动预热。

## 2. 目录结构

| 路径 | 作用 |
|------|------|
| `data/wechat/<会话>/messages.jsonl` | 每行一条消息（ts/role/sender/text） |
| `data/wechat/<会话>/chat.txt` | 按时间排序的可读记录（纯文本） |
| `data/raw/wechat/<会话>.md` | 给 `ingest.py` 用的原料 |
| `data/knowledge_store.json` | RAG 向量库 |
| `data/wechat/<会话>/vectors.json` | 该会话历史检索的向量索引（增量更新） |
| `data/profiles/<会话>.md` | 联系人/群画像（性格、关系、态度、经历） |
| `data/entity_notes.json` | 本地地名/梗备注（优先于搜索） |
| `data/finetune/sft_all*.jsonl` | 微调数据（全量 / train / val） |
| `data/finetune/sft*.jsonl` | 只学“我”的数据 |
| `training/models/` | 基础模型（HF safetensors） |
| `training/output-all/` | 全量 LoRA 适配器 |
| `training/merged-all/` | 合并后的完整模型 |
| `training/merged-all-q8.gguf` | 导入 Ollama 的 GGUF |
| `config.json` | 模型/端口/知识库配置 |

## 3. 数据采集

### 3.1 为什么用“聊天记录”窗口

- 聊天框里只有气泡，需要按颜色猜发送者，且**没有精确时间**
- 历史记录窗口是列表：**发送者（左，灰）+ 精确时间（右，灰）+ 内容（下）**，解析最准

### 3.2 采集流程（`scripts/wechat_collect_history.py`）

```
找到“聊天记录”窗口
  → 拉高窗口（宽度不变，避免 OCR 缩图丢字）
  → 分段 OCR（每段 700px）
  → 解析：发送者 / 时间 / 内容
  → 每屏向下滚 N 次（默认 18 次小滚轮）
  → 合并去重 → 写入 messages.jsonl / chat.txt / raw md
  → 还原窗口大小
```

关键处理：

| 处理 | 原因 |
|------|------|
| 发送者在内容**上方 5–60px** | 修正过配对方向反了的问题 |
| 没有发送者标签 → 群聊丢弃 | 接龙/系统内容会回退成群名，污染语料 |
| 过滤 `1. 2. 3.` 编号行 | 接龙名单不是聊天 |
| 过滤长度 ≤2 的数字/符号碎片 | OCR 残片 |
| 群名去括号 | `小鸽哥踢球群（112）` → `小鸽哥踢球群` |
| 过滤纯 @ 名单（`@a@b@c`） | 避免模型学成 @ 刷屏 |
| 清除"我资料里没有"等拒绝语 | 机器人自己说过的话被采集回来，会让模型学会拒绝 |

### 3.3 命令

```powershell
# 采集（先在微信打开目标群的“聊天记录”窗口）
python scripts\wechat_collect_history.py --screens 10 --step 18

# 自动打开历史窗口（点聊天标题栏的“聊天气泡 ▾”）
python scripts\wechat_collect_history.py --chat 小鸽哥踢球群 --auto-open
```

### 3.4 语料清洗（`scripts/clean_corpus.py`）

```powershell
python scripts\clean_corpus.py            # 只统计
python scripts\clean_corpus.py --apply    # 写回
```

清洗规则：

| 规则 | 原因 |
|------|------|
| 去掉系统消息（撤回/入群/拍一拍/红包…） | 不是对话 |
| 去掉纯 @ 名单 | 会让模型学成 @ 刷屏 |
| 去掉低信息短句（嗯/哦/哈/6/收到…） | 拉低回复质量 |
| 去掉纯表情/符号/链接/单字符 | 没有学习价值 |
| 近期重复去重 | OCR 重叠采集 |
| **人名归一**（张驰→张弛、小江、→小江、Richer 变体→Richer…） | 按人训练前必须合并同一人 |

> 清洗后：2278 → **2061 条**。

## 4. RAG（检索增强）

### 4.1 原理

1. **切块**：把 `data/raw/**` 按 500 字切块，块间重叠 80 字
2. **嵌入**：每块用 `bge-m3` 转成向量，存 `knowledge_store.json`
3. **检索**：问题也转向量，按余弦相似度取 top-k
4. **回答**：把检索片段拼进 prompt，让模型**只依据资料**回答

### 4.2 命令

```powershell
python scripts\ingest.py                 # 重建向量库
python scripts\query.py "闫总是谁"        # 检索 + 回答
```

> 注意：`ingest.py` 每次会**重置**向量库并重新嵌入全部 `data/raw`。

## 5. 微调（LoRA / QLoRA）

### 5.1 数据格式

每行一条样本：

```json
{"messages":[
  {"role":"system","content":"你正在扮演「张博」。场景：微信群「小鸽哥踢球群」，正在和「王涵」等人聊天。请按他/她平时的说话方式回复。"},
  {"role":"user","content":"王涵：今天来几个\n张博：我可能晚点到"},
  {"role":"assistant","content":"行，我先占个位"}
]}
```

- `system`：**扮演谁 + 什么场合 + 跟谁聊**（场景化，让模型学会“不同人不同场合”）
- `user`：最近 6 条上下文（带发送者名）
- `assistant`：目标人物的真实回复

### 5.2 两个关键设计

| 设计 | 作用 |
|------|------|
| **assistant-only loss** | 只对回复算 loss；否则模型会背上下文，输出“田政：红烧肉”这种整段对话 |
| **`{% generation %}` 模板** | 让 trl 精确定位回复段（含 `<|im_end|>`，学会停） |

> **SYSTEM 提示词说明**：`training/*.Modelfile` 里的 `SYSTEM` 只对 `ollama run` 生效；
> 经 `agent.chat_reply` 调用时会被 `persona.md` + 画像覆盖，所以 Modelfile 里不再写死。
> 训练时的 system 是「你正在扮演「X」。场景：…」，推理时是 persona + 硬规则，
> 两者不完全一致（已知问题，后续可统一重建）。

### 5.3 防过拟合参数

| 参数 | 值 | 说明 |
|------|----|------|
| epochs | 2 | 1 太少欠拟合，3+ 容易背句子 |
| lr | 1.2e-4 | 介于欠拟合(1e-4)与过拟合(1.5e-4)之间 |
| 最优 epoch | `load_best_model_at_end` | 按 eval loss 自动选最好的 epoch，避免 epoch 2 变差 |
| LoRA r / alpha | 16 / 32 | 容量适中 |
| LoRA dropout | 0.1 | 抑制记忆 |
| weight decay | 0.01 | 抑制过大权重 |
| 调度 | cosine + warmup 10 | 平滑收敛 |
| max_length | 512 | 对话样本够用 |

**判断过拟合**：train loss 持续降、eval loss 反而升，就减 epoch 或加数据。

### 5.4 命令

```powershell
# 1) 导出数据
python scripts\export_sft.py --all
# 按轮次导出（合并连续同人、不标发言人、带时间间隔）
python scripts\export_sft.py --turns

# 2) 训练（16G 显存：7B + 4bit QLoRA）
python training\train_lora.py --model training\models\Qwen2.5-7B-Instruct --epochs 2 --lr 1.2e-4

# 3) 合并
python training\merge_lora.py --model training\models\Qwen2.5-7B-Instruct --adapter training\output-all --output training\merged-all

# 4) 转 GGUF（q8_0，约 8GB）
python training\convert_hf_to_gguf.py training\merged-all --outfile training\merged-all-q8.gguf --outtype q8_0

# 5) 导入 Ollama
ollama create clone-richer-v3 -f training\Modelfile
```

> 为什么 q8_0 而不是 q4：转换脚本只支持 f16/bf16/q8_0；q8 质量好、16G 显存放得下（8GB）。

### 5.5 训练记录（loss 平衡过程）

| 版本 | 数据 | 参数 | eval loss | 现象 |
|------|------|------|-----------|------|
| v2 | 1140 | 1 epoch, lr 1e-4 | 4.57 | 欠拟合，回复弱 |
| v3 | 1874 | 2 epoch, lr 1.5e-4 | 4.09 / 4.15 | epoch 2 略过拟合 |
| v4 | 2255 | 2 epoch, lr 1.2e-4, 选最优 epoch | **3.905**（epoch 1） | 平衡版，loss 最好 |

结论：

- lr 1.2e-4 + 按 eval loss 选最优 epoch，能把 loss 压到 3.9 且不继续过拟合
- 但**回复仍偏“回声”**（问“你谁啊”答“你谁啊”），瓶颈在数据质量：
  - 群里大量短句、@、重复接龙，模型学到的“回复”本身信息量低
  - 想明显提升，需要 5000+ 条干净语料，或**按人/按群分别训练**

### 5.6 按人训练（每人一个 LoRA）

目标：让分身能**按指定人的风格**说话。

```powershell
# 1) 按人导出（默认 ≥100 条才导出）
python scripts\export_person.py --min-samples 100

# 2) 训练某个人（自动完成：训练 → 合并 → GGUF → Ollama）
python scripts\person_pipeline.py --person 王宇航

# 3) 或训练所有达标的人
python scripts\person_pipeline.py --all --min-samples 100
```

模型命名：中文名映射为 ASCII，例如 `王宇航 → clone-wangyuhang`、`田政 → clone-tianzheng`。

当前达标的人（≥100 条）：

| 人 | 样本 | 模型 |
|----|------|------|
| 王宇航 | 310 | clone-wangyuhang |
| 田政 | 248 | clone-tianzheng |
| 小江 | 214 | clone-xiaojiang |
| 苏博 | 189 | clone-subo |
| 盖瑞 | 187 | clone-gairy |
| 马灯 | 148 | clone-madeng |
| 闫啸SEAN | 140 | clone-sean |

> 每人样本 100–300 时，2 epoch + lr 1.2e-4 即可；人少时把 `--epochs` 降到 1。
> 想“一个模型切换多人”，用 `sft_all`（system 指定扮演谁）；想“每个人最像”，用按人训练。

## 6. 模型接入

`config.json`：

```json
{
  "ollama": { "chat_model": "qwen3.5:9b" },
  "chat_reply_model": "clone-richer-v3"
}
```

- **回复生成**（`chat_reply`）优先用 `clone-richer-v3`
- **分类 / 工具 / RAG** 仍用 `qwen3.5:9b`（微调模型不一定遵守 JSON 输出）

### 6.1 三层路由

| 场景 | 用哪个模型 |
|------|-----------|
| 会话在 `config.json` 的 `chat_models` 里指定了模型 | 该人格模型（如 `clone-wangyuhang`） |
| 会话在 `data/wechat/<会话>` 有语料 | 微调模型（学口吻） |
| 会话没有对应语料 | 基础模型 `qwen3.5:9b`（通用能力） |
| 微调输出空 / 拒绝语 / @刷屏复读 | 自动回退基础模型 |

示例（按群/按人指定不同人格）：

```json
"chat_models": {
  "外卖一群": "clone-wangyuhang",
  "小鸽哥踢球群": "clone-richer-v3"
}
```

### 6.2 回复后处理

- **拒绝语兜底**：命中"资料里/资料库/我不记得"→ 换基础模型重答
- **复读折叠**：连续重复片段只留一次
- **@ 限制**：最多保留 2 个 @
- **判为复读**（>120 字或 >4 个 @）→ 换基础模型

改完重启 `run_ui.bat` 生效。

## 7. 一键流水线

### 7.1 手动

```powershell
# 只更新数据 + RAG（几十秒）
python scripts\auto_pipeline.py

# 全量：导出 + RAG + 重训 + 合并 + 转 GGUF + 导入（约 25 分钟）
python scripts\auto_pipeline.py --train

# 或双击
run_pipeline.bat --train
```

### 7.2 定期重训（Windows 计划任务）

```powershell
# 每周日 03:00 自动重训
schtasks /Create /SC WEEKLY /D SUN /TN "AI_Clone_Retrain" ^
  /TR "E:\Python\Python311\python.exe E:\agent_engineer\AI_clone\scripts\auto_pipeline.py --train" ^
  /ST 03:00
```

> 采集本身需要微信窗口，建议手动采集后跑流水线；自动任务负责“导出 + 重训”。

## 8. 参数速查

| 脚本 | 常用参数 |
|------|---------|
| `wechat_collect_history.py` | `--screens 10 --step 18 --auto-open --chat 群名` |
| `export_sft.py` | `--all`（所有人）/ 默认只学我 / `--split 0.9` |
| `clean_corpus.py` | `--apply` 真正清洗；默认只统计 |
| `export_person.py` | `--min-samples 100` |
| `person_pipeline.py` | `--person 王宇航` / `--all` |
| `train_lora.py` | `--model` `--epochs` `--lr` `--max-length` `--no-4bit` |
| `merge_lora.py` | `--model` `--adapter` `--output` |
| `auto_pipeline.py` | `--train` `--epochs` `--lr` `--skip-ingest` |
| `ingest.py` / `query.py` | `query.py "问题" --topk 6` |

## 9. 常见问题

| 现象 | 原因 / 解决 |
|------|------------|
| `no kernel image is available` | RTX 50 系需要 cu128 版 PyTorch |
| `pip` 报 GBK 解码错误 | requirements 里有中文注释；已改为纯 ASCII |
| hf-mirror SSL 断开 | 改用 `modelscope download` 下模型 |
| `ollama create` 报 MLX | Ollama 0.40 不支持 HF safetensors，先转 GGUF |
| 显存不足 | `ollama stop` 释放；或换 3B / `--no-4bit` |
| 回复在背聊天记录 | 检查是否开了 assistant-only loss |
| 回复总是“@某人” | 数据里 @ 太多；清洗后再训 |
| 回复总说“资料里没有” | persona/query 提示词已改 + 回复兜底；彻底根除需重训 |
| 回复 @ 刷屏 | 已过滤纯 @ 名单、复读折叠、超限回退基础模型 |
| 采集不到消息 | 历史窗口没开 / 发送者标签缺失被过滤 |

## 10. 当前数据规模（会变化）

| 项 | 数量 |
|----|------|
| 微信消息 | 约 2745 条 / 8 个会话（外卖一群 846、小鸽哥 694、王宇航 376、李志杰 231、冯自灏 186、敖民 150、🐦🐦 242、苏博 20） |
| SFT 样本（全量） | **2621 条**（train 2358 / val 263） |
| SFT 样本（只学我） | 477 条（train 429 / val 48） |
| 画像 | 4 个（外卖一群、小鸽哥踢球群、王宇航、🐦🐦） |
| RAG 片段 | 约 61 条（小鸽哥踢球群） |

## 11. 变更记录

### 2026-10-25

| 变更 | 说明 |
|------|------|
| 回复模型切换 | `clone-richer-v3`（复读）→ 基础模型 + 画像 + 历史检索 + 风格示例 |
| 提速 | 原生 `/api/chat`、`think:false`、`num_ctx 8k`、`keep_alive 30m`、启动预热；回复从 30–70s 降到 ~1.2s |
| 联系人画像 | `build_profiles.py` 生成性格/关系/态度/经历；`chat_reply` 注入 |
| 历史检索 | `wechat_history_rag.py` 按会话向量检索 top-3；增量索引 |
| 意图分析 | 增加 `topic`、`调侃/玩笑`、`addressed_to_me`（含“这个AI/学长”） |
| 邀约规则 | 群内邀约未点名我 → 不回复（分类阶段拦截） |
| 名词速查 | `entity_notes.json` 优先，Bing 2 秒兜底；调侃类不速查 |
| 风格后处理 | 书面化短语替换；删除主动约时间/提议聚聚的句子 |
| 训练 | 全量 2621 条重训（2 epoch，lr 1.2e-4，最优 epoch 选择）→ `output-all` |
| 视觉 | `qwen2.5vl:3b-ctx` 读红点行；红点加“头像外部+实心圆”双约束 |
| 7B VLM | 已下载/训练/合并（`training/vlm-merged`），GGUF 转换待继续 |

## 12. 脚本与启动程序（按步骤）

### 12.1 启动 / 日常使用（根目录 .bat）

| 文件 | 作用 | 用法 |
|------|------|------|
| `run_ui.bat` | **推荐入口**：打开控制台（配置白名单、启动/暂停/停止、采集、日志） | 双击 |
| `run_bot.bat` | 直接启动多聊机器人（不开界面） | 双击 |
| `stop_bot.bat` | 停止机器人（写 `STOP` 标志） | 双击 |
| `run_dry.bat` | 干跑一轮：只判断、不发送 | 双击 |
| `run_reply_now.bat` | 立即回复一次（单聊，`--once`） | 双击 |
| `run_pipeline.bat` | 一键流水线（导出 + RAG；加 `--train` 重训） | `run_pipeline.bat --train` |

### 12.2 数据采集

| 脚本 | 作用 | 典型用法 |
|------|------|---------|
| `wechat_collect_history.py` | **主采集**：从“聊天记录”窗口逐屏 OCR（发送者/时间/内容） | `python scripts\wechat_collect_history.py --screens 10 --step 18 --auto-open --chat 小鸽哥踢球群` |
| `wechat_historian.py` | 旧方式：在当前聊天框里**向上滚动**采集 | `python scripts\wechat_historian.py --screens 5 --step 5` |
| `wechat_collect.py` | 聊天框**向下翻页**采集（无历史窗口时备用） | `python scripts\wechat_collect.py --screens 10 --step 3` |
| `wechat_capture.py` | 截微信窗口 + 全窗 OCR，用于**校准坐标** | `python scripts\wechat_capture.py` |

### 12.3 语料清洗 / 画像

| 脚本 | 作用 | 用法 |
|------|------|------|
| `clean_corpus.py` | 清洗：去系统消息/纯@/短句/重复，人名归一 | `python scripts\clean_corpus.py`（统计）<br>`python scripts\clean_corpus.py --apply`（写回） |
| `build_profiles.py` | 生成联系人/群画像（性格/关系/态度/经历） | `python scripts\build_profiles.py`<br>`python scripts\build_profiles.py --chat 外卖一群` |

### 12.4 RAG（检索增强）

| 脚本 | 作用 | 用法 |
|------|------|------|
| `ingest.py` | 把 `data/raw/**` 切块 + bge-m3 嵌入，重建知识库 | `python scripts\ingest.py` |
| `query.py` | 检索 + 让模型只依据资料回答 | `python scripts\query.py "闫总是谁" --topk 6` |

### 12.5 微调

| 脚本 | 作用 | 用法 |
|------|------|------|
| `export_sft.py` | 导出 SFT 数据（默认只学“我”，`--all` 所有人，`--turns` 按轮次） | `python scripts\export_sft.py --all` |
| `export_person.py` | 按人导出（每人一份 train/val） | `python scripts\export_person.py --min-samples 100` |
| `training/train_lora.py` | QLoRA 训练（assistant-only、防过拟合） | `python training\train_lora.py --data data\finetune\sft_all_train.jsonl --val ... --output training\output-all --epochs 2 --lr 1.2e-4` |
| `training/merge_lora.py` | 合并 LoRA 到基础模型 | `python training\merge_lora.py --adapter training\output-all --output training\merged-all` |
| `training/convert_hf_to_gguf.py` | 转 GGUF（q8_0） | `python training\convert_hf_to_gguf.py training\merged-all --outfile training\merged-all-q8.gguf --outtype q8_0` |
| `training/train_vlm_lora.py` | VLM（Qwen2.5-VL）LoRA 训练 | `python training\train_vlm_lora.py --epochs 1 --lr 1e-4` |
| `training/merge_vlm_lora.py` | 合并 VLM 适配器 | `python training\merge_vlm_lora.py --adapter training\vlm-output --output training\vlm-merged` |
| `person_pipeline.py` | 按人训练一条龙（导出→训练→合并→GGUF→Ollama） | `python scripts\person_pipeline.py --person 王宇航`<br>`python scripts\person_pipeline.py --all --min-samples 100` |
| `auto_pipeline.py` | 一键流水线（数据+RAG；`--train` 重训） | `python scripts\auto_pipeline.py --train` |

### 12.6 微信机器人

| 脚本 | 作用 | 用法 |
|------|------|------|
| `wechat_multi.py` | **多聊自动回复**：红点检测→白名单→分类→意图→回复→收起 | `python scripts\wechat_multi.py --live` |
| `wechat_screener.py` | 单聊自动回复（按目标会话） | `python scripts\wechat_screener.py --live --target 杨文祥` |
| `wechat_ui.py` | Tkinter 控制台（后台调用上面两个） | `python scripts\wechat_ui.py` |

### 12.7 调试 / 自检

| 脚本 | 作用 | 用法 |
|------|------|------|
| `check.py` | 环境自检：Ollama、模型、配置 | `python scripts\check.py` |
| `ocr_test.py` | 对一张图片做 OCR 并打印结果 | `python scripts\ocr_test.py 图片.png` |
| `_*.py` | 临时探针（红点/头像/分类测试），可删 | — |

> 日常只需要：`run_ui.bat` 启动；采集用 `wechat_collect_history.py`；更新语料后跑 `auto_pipeline.py`。

## 13. 实时采集与跟聊（P0）

### 13.1 实时入库

机器人每次 OCR 到消息后，**先写入语料库再决定是否回复**：

```
OCR → build_history → append_messages（去重）→ export_text → 分类/回复
```

- 去重键：`(ts, sender, text)`，没有时间戳时退化为 `(sender, text)`
- 新消息带 `source: "live"`，与历史采集的 `collect` 区分
- 只在新增加时导出 `chat.txt`，不重复写盘

### 13.2 跟聊循环

回复后不再立即收起，而是：

```
发送 → 等 3 秒 → 重新 OCR
  ├─ 有新消息 → 继续回复（最多 3 轮）
  └─ 连续 2 次无新消息 → 收起聊天
```

- 最多 3 轮，防止和人"对刷"
- 跟聊期间也会实时入库
- 新消息判定：`最后一条 incoming 时间 > 最后一条 outgoing 时间`

### 13.3 检索窗口

`history` 检索只取：

```json
"history_days": 3,
"history_max_messages": 999
```

- 先按 `ts` 过滤最近 3 天（无时间戳的保留）
- 超过 999 条只取最近 999
- 再做余弦 top-k（`history_topk` / `history_min_score` / `history_window`）

> 效果：不会因为几个月前的旧消息干扰当前话题，也避免索引无限增长后检索变慢。

### 13.4 轮次合并与时间间隔

训练与推理都按“轮次”处理：

- **合并**：连续同一发送者的多段话合并成一轮
- **训练**：`python scripts\export_sft.py --turns` → `sft_turns_*.jsonl`（不标注发言人）
- **推理**：`chat_reply` 先把连续同角色消息合并，再拼上下文
- **时间间隔**：相邻轮次间隔 > `gap_threshold_minutes`（默认 30）时，提示“可能是新话题，不要强行衔接”

### 13.5 情绪 / 潜台词 / 玩梗

意图分析新增三个字段，并注入回复提示：

| 字段 | 说明 | 示例 |
|------|------|------|
| `emotion` | 开心/调侃/抱怨/试探/炫耀/认真/敷衍 | 试探 |
| `subtext` | 潜台词，一句话；没有为空 | 可能在寻求安慰或认可 |
| `meme` | 是否在玩梗/引用 | 用 2 块钱把 AI 踢下线 |

```json
{"emotion":"试探","subtext":"可能在寻求安慰或认可","meme":""}
```

### 13.6 当前会话免点击 / 人名校验

- **免点击**：点击前先 OCR 标题栏；当前已是目标会话则不再点击，避免把聊天误关
- **人名校验**：回复里出现“叫上/找/问/跟 + 人名”，但该名字不在历史发送者/我的昵称里 → 判定为编造
  1. 重新生成一次（提示不要提该名字）
  2. 仍出现则删掉含该名字的分句
