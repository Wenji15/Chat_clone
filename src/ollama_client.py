"""Ollama 客户端封装。

只依赖 Python 标准库（urllib），通过 Ollama 的 OpenAI 兼容接口调用：
  - 对话：POST /v1/chat/completions（支持 function calling）
  - 嵌入：POST /v1/embeddings

这样整个项目不需要安装 openai / requests / langchain 等任何第三方包。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    """Ollama 连接或响应异常。"""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        chat_model: str = "qwen2.5:7b",
        embed_model: str = "nomic-embed-text",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaError(
                f"无法连接 Ollama（{self.base_url}）。\n"
                "请先安装并启动 Ollama，再拉取模型，例如：\n"
                f"  ollama pull {self.chat_model}\n"
                f"  ollama pull {self.embed_model}\n"
                f"原始错误：{e}"
            ) from e
        except json.JSONDecodeError as e:
            raise OllamaError(f"Ollama 返回了无法解析的内容：{e}") from e

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        model: str | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        """发送对话，返回 assistant 消息（含 content 与/或 tool_calls）。

        model 为空时用默认 chat_model；可用于让“回复生成”走微调模型，
        而分类/工具调用仍用基础模型。
        think 为空时用 self.think_default（配置 thinking_answer）。
        """
        think_val = think if think is not None else getattr(self, "think_default", None)
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
        }
        if think_val is not None:
            payload["think"] = bool(think_val)
        if tools:
            payload["tools"] = tools
        if max_tokens:
            payload["max_tokens"] = max_tokens

        result = self._post("/v1/chat/completions", payload)
        try:
            return result["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise OllamaError(
                "Ollama 对话响应结构异常："
                f"{json.dumps(result, ensure_ascii=False)[:500]}"
            ) from e

    # ------------------------------------------------------------------ #
    def chat_native(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        think: bool = False,
        num_ctx: int = 32768,
        keep_alive: str = "30m",
    ) -> str:
        """Ollama 原生 /api/chat：可关闭 thinking、限制上下文、保持常驻。"""
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }
        result = self._post("/api/chat", payload)
        return str((result.get("message") or {}).get("content", ""))

    # ------------------------------------------------------------------ #
    def chat_vision(
        self,
        prompt: str,
        images_b64: list[str],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """用视觉模型看图并返回文本（Ollama 原生 /api/chat，images 为 base64）。"""
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": [
                {"role": "user", "content": prompt, "images": list(images_b64)}
            ],
            "stream": False,
            "options": {"temperature": temperature},
            "keep_alive": "30s",
        }
        result = self._post("/api/chat", payload)
        return str((result.get("message") or {}).get("content", ""))

    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量把文本转成向量，返回与输入顺序一致的向量列表。"""
        payload = {"model": self.embed_model, "input": list(texts)}
        result = self._post("/v1/embeddings", payload)
        try:
            items = sorted(result["data"], key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in items]
        except (KeyError, TypeError) as e:
            raise OllamaError(
                "Ollama 嵌入接口响应异常（请确认已执行 "
                f"`ollama pull {self.embed_model}`）："
                f"{json.dumps(result, ensure_ascii=False)[:500]}"
            ) from e
