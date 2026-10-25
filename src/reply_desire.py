"""回复欲望分类（规则 4）。

A 直接对我（@我/点名/引用我）→ 必回
B 与我相关（涉及我的工作/兴趣）→ 分数过阈值才回
C 无关但可参与（轻松话题/有人提问）→ 高阈值、低概率搭话
D 完全无关（别人私事/刷屏/广告/纯表情）→ 不回

实现：先用规则抓 A，其余交给本地模型分类打分。
"""
from __future__ import annotations

import json
import re

CATEGORY_DESC = {
    "A": "直接对我",
    "B": "与我相关",
    "C": "无关但可参与",
    "D": "完全无关",
}


def rule_hit(text: str, my_names: list[str]) -> bool:
    """规则层：文本里出现 @我 或我的名字/昵称。"""
    t = text or ""
    for name in my_names or []:
        if name and name in t:
            return True
    return False


def classify(
    agent,
    text: str,
    context: str,
    my_names: list[str],
    interests: list[str],
) -> dict:
    """返回 {"class": "A|B|C|D", "score": 0.0-1.0, "reason": str}。"""
    if rule_hit(text, my_names):
        return {
            "class": "A",
            "score": 1.0,
            "reason": "规则命中：@我/点名",
            "invitation": False,
            "addressed_to_me": True,
        }

    interests_str = "、".join(interests) if interests else "（未设置）"
    system = (
        "你在判断一条微信群消息该不该由我回复。只输出 JSON，不要任何解释。\n"
        f"我的名字/昵称：{'、'.join(my_names) if my_names else '未知'}\n"
        f"我在意的话题：{interests_str}\n"
        "分类标准：\n"
        "A 直接对我：@我、点名、引用我的消息；\n"
        "B 与我相关：话题涉及我在意的事，值得回应；\n"
        "C 无关但可参与：轻松闲聊、有人在提问，可以搭话；\n"
        "D 完全无关：广告、刷屏、纯表情、明显与我无关的私事。\n"
        "注意：有人在提问、在聊群内共同话题（球/麻将/吃喝）、在调侃我，"
        "都不算 D，至少 C。\n"
        "另外判断：invitation 表示这条消息是否在约人/组局"
        "（约球、约麻将、约饭等）；addressed_to_me 表示是否明确点名邀请我本人。\n"
        '输出格式：{"class":"A|B|C|D","score":0.0-1.0,"reason":"一句话",'
        '"invitation":true或false,"addressed_to_me":true或false}'
    )
    user = f"最近对话：\n{context}\n\n新消息：{text}"

    content = agent.llm.chat_native(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    ).strip()
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            cls = str(data.get("class", "D")).strip().upper()[:1]
            if cls not in "ABCD":
                cls = "D"
            score = float(data.get("score", 0) or 0)
            return {
                "class": cls,
                "score": max(0.0, min(1.0, score)),
                "reason": str(data.get("reason", ""))[:120],
                "invitation": bool(data.get("invitation", False)),
                "addressed_to_me": bool(data.get("addressed_to_me", False)),
            }
        except Exception:
            pass
    return {
        "class": "D",
        "score": 0.0,
        "reason": f"解析失败：{content[:80]}",
        "invitation": False,
        "addressed_to_me": False,
    }


def should_reply(result: dict, thresholds: dict) -> bool:
    """根据分类结果和阈值决定要不要回。"""
    cls = result.get("class", "D")
    score = float(result.get("score", 0))
    if cls == "A":
        return True
    if cls == "B":
        return score >= float(thresholds.get("B", 0.6))
    if cls == "C":
        return score >= float(thresholds.get("C", 0.75))
    return False
