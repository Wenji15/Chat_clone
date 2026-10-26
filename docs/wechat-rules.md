# 微信助手 · 8 条规则实施状态

| # | 规则 | 状态 | 对应实现 |
|---|------|------|---------|
| 1 | 目标人物索引 / 针对性语气 | 🚧 基础已建 | [contacts.json](../contacts.json) + 会话标题识别 |
| 2 | 每个会话一个文件，存发送人/内容/时间 | ✅ 已完成 | [src/wechat_store.py](../src/wechat_store.py) → `data/wechat/<会话>/messages.jsonl` + `chat.txt` |
| 3 | 自动滚动采集历史对话 | ✅ 已完成 | [scripts/wechat_collect.py](../scripts/wechat_collect.py)（带发送者/时间） |
| 4 | ~~回复欲望 A/B/C/D~~ | ❌ 已取消 | 统一白名单，不再分类 |
| 5 | 自动识别未读红点 | ✅ 已实现 | [src/wechat_unread.py](../src/wechat_unread.py)（大徽标 + 小红点） |
| 6 | 发完 3 秒切回列表→找未读→进对话框 | ✅ 已实现 | [scripts/wechat_multi.py](../scripts/wechat_multi.py)（白名单） |
| 7 | 识别频率可控（3 秒一次） | ✅ 已完成 | `bot_config.json` 的 `interval` |
| 8 | 简易可视化界面 | ✅ 已完成 | [scripts/wechat_ui.py](../scripts/wechat_ui.py)（Tkinter 控制台，含设置编辑） |

## 已落地的数据结构

```
data/wechat/
  <会话名>/
    messages.jsonl   # {"ts","role":"me|them","sender","text"}
    chat.txt         # 可读导出（纯文本）
```

## 使用

```powershell
# 采集当前打开的会话历史（向上翻 5 屏）
python scripts/wechat_historian.py --screens 5 --step 5
```

前置：微信停在目标聊天窗口。

## 已知限制

- 历史采集靠滚动+OCR，会有错误/遗漏；图片、表情、语音无文字。
- 时间戳只能近似（微信只在部分消息显示时间）。
- 群聊发送者靠灰色名字标签识别。
- 白名单统一管理单聊/群聊，不再做回复欲望分类。
