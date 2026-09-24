"""Small Responses API adapter. No SDK dependency and no implicit API calls."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rag.core import RAG_INSTRUCTIONS


class ModelServiceError(RuntimeError):
    """Only controlled, credential-free messages may cross the HTTP boundary."""


class OpenAIProvider:
    name = "openai"
    embedding_model = None  # Retrieval remains local in this integration.

    def __init__(self, generation_model=None, api_key=None, max_output_tokens=1024):
        self.generation_model = generation_model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        self._api_key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        if not self._api_key:
            raise ValueError("请设置 OPENAI_API_KEY，或使用 --prompt-api-key 隐藏输入密钥。")
        if type(max_output_tokens) is not int or not 64 <= max_output_tokens <= 4096:
            raise ValueError("max_output_tokens must be an integer between 64 and 4096")
        self.max_output_tokens = max_output_tokens

    def embed(self, texts):
        raise ValueError("OpenAI 模式当前使用本地 BM25 检索，不支持 dense/hybrid。")

    def generate(self, question, evidence):
        # Do not send unrelated corpus chunks, retrieval scores or local file paths.
        prompt = json.dumps({"question": question, "evidence": [
            {k: c[k] for k in ("id", "title", "text")} for c in evidence
        ]}, ensure_ascii=False)
        if len(prompt) > 24000:
            raise ValueError("证据超过单次 24000 字符限制，请减小资料块或检索数量。")
        payload = {"model": self.generation_model, "instructions": RAG_INSTRUCTIONS,
                   "input": prompt, "store": False, "stream": False,
                   "max_output_tokens": self.max_output_tokens}
        request = Request("https://api.openai.com/v1/responses",
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self._api_key,
                                   "Content-Type": "application/json"})
        try:
            # No automatic retries: a network timeout can have incurred usage already.
            with urlopen(request, timeout=90) as response:
                data = json.load(response)
        except HTTPError as exc:
            messages = {
                400: "OpenAI 请求参数不被接受，请检查所选模型和输出限制。",
                401: "OpenAI API 密钥无效或已失效，请在本机更新密钥。",
                403: "OpenAI 拒绝访问，请检查项目、模型权限及服务可用性。",
                404: "OpenAI 模型不存在或当前项目无权访问，请检查模型名称。",
                429: "OpenAI 请求受限，请检查 API 额度、账单和速率限制。",
            }
            message = messages.get(exc.code, f"OpenAI 服务返回 HTTP {exc.code}，请稍后手动重试。")
            exc.close()
            raise ModelServiceError(message) from None
        except (URLError, TimeoutError, OSError):
            raise ModelServiceError("无法连接 OpenAI 或请求超时；未自动重试，请检查网络。") from None
        except (ValueError, UnicodeError):
            raise ModelServiceError("OpenAI 返回了无法解析的响应。") from None

        if not isinstance(data, dict) or data.get("status") != "completed":
            raise ModelServiceError("OpenAI 未完整生成回答，可能达到输出限制；未展示截断内容。")
        texts = []
        try:
            for item in data.get("output", []):
                if item.get("type") != "message" or item.get("role") != "assistant":
                    continue
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise ModelServiceError("模型拒绝了本次请求。")
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
        except (AttributeError, TypeError):
            raise ModelServiceError("OpenAI 返回了非预期的回答格式。") from None
        answer = "\n".join(texts).strip()
        if not answer:
            raise ModelServiceError("OpenAI 未返回可展示的文本。")
        usage = data.get("usage") or {}
        safe_usage = {k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")
                      if type(usage.get(k)) is int} if isinstance(usage, dict) else {}
        return answer, safe_usage
