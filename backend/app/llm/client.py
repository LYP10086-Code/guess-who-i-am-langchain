"""LLM 客户端（LangChain 1.0）：加载根目录 .env，用标准工厂 init_chat_model 构造模型。

提供商通过 LLM_PROVIDER 选择（缺省 anthropic），create_agent 只依赖此处返回的
标准 BaseChatModel，不感知具体厂商：

- anthropic（默认）：读 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL / MODEL_ID，
  兼容 Anthropic 官方与国内 Anthropic 兼容端点；
- openai：读 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_ID，
  可接 OpenAI 官方与各类 OpenAI 兼容端点；
- deepseek：读 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL（可选）/ MODEL_ID，
  经 langchain-deepseek 官方集成接入，默认关闭 thinking。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# backend/app/llm/client.py → parents[3] = 项目根目录（.env 所在处）
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATH = _REPO_ROOT / ".env"

if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH, override=False)

# 与旧版一致的默认生成长度上限（langchain-anthropic v1 默认值有变化，显式锁定）
DEFAULT_MAX_TOKENS = 2000

_VALID_PROVIDERS = ("anthropic", "openai", "deepseek")


def get_provider() -> str:
    """模型提供商：anthropic（默认）/ openai / deepseek。"""
    provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    return provider if provider in _VALID_PROVIDERS else "anthropic"


def get_model() -> str:
    return os.getenv("MODEL_ID", "claude-sonnet-4-6")


def get_chat_model(*, max_tokens: int = DEFAULT_MAX_TOKENS):
    """按当前环境变量，用 init_chat_model 标准工厂构造 ChatModel（无状态、可重复创建）。"""
    provider = get_provider()
    model = get_model()

    if provider == "deepseek":
        kwargs: dict = {
            "model": model,
            "model_provider": "deepseek",
            # DeepSeek 思考型模型：显式关闭 thinking，保证二元问答/工具调用稳定
            "extra_body": {"thinking": {"type": "disabled"}},
            "max_tokens": max_tokens,
        }
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url
        return init_chat_model(**kwargs)

    if provider == "openai":
        kwargs = {"model": model, "model_provider": "openai", "max_tokens": max_tokens}
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if api_key:
            kwargs["openai_api_key"] = api_key
        if base_url:
            kwargs["openai_api_base"] = base_url
        return init_chat_model(**kwargs)

    # anthropic（默认）
    base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("ANTHROPIC_API_URL") or None
    if base_url:
        # 兼容国内 Anthropic 兼容端点（避免误用 SDK 旧 token 变量）
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    kwargs = {"model": model, "model_provider": "anthropic", "max_tokens": max_tokens}
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        kwargs["anthropic_api_key"] = api_key
    if base_url:
        kwargs["anthropic_api_url"] = base_url
    return init_chat_model(**kwargs)
