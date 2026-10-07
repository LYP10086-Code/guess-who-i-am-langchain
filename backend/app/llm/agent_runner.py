"""Agent Loop（LangChain 1.0 / create_agent 版）：CLI / Web 共用。

run_agent_loop(session, messages, system_prompt=...)：
- 将 session 绑定到 contextvar，工具 handler 通过 get_session() 取得；
- 使用 LangChain 1.0 create_agent（底层 LangGraph）执行标准循环：
  模型 → 工具调用 → 工具结果回灌 → … → 无工具调用即结束；
- PhaseGuardMiddleware 复刻旧版三道关键语义：
  1. 阶段白名单：越权工具不执行 handler、状态零修改，回灌 [ERROR]；
  2. 同轮多个 tool_call 串行执行（对齐旧手写 for 循环，避免并发改状态机）；
  3. 工具异常归一为 [ERROR] 文本并触发 PostToolUse hook；
- 无工具调用即结束，返回最后一条 assistant 文本（host_message）。
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from game_state import (
    set_current_session,
    reset_current_session,
    GameStatus,
)
from hooks import trigger_hooks
from tools import (
    LANGCHAIN_TOOLS,
    is_tool_allowed,
    tool_denied_message,
)

from .client import get_chat_model


class AgentLoopError(RuntimeError):
    pass


class UpstreamLLMError(AgentLoopError):
    """LLM 超时/限流/返回异常；调用方保证状态未落账。"""


def _force_any_tool_choice(model) -> object:
    """各厂商"必须调用某个工具"的 tool_choice 取值（跨提供商兼容）。"""
    try:
        from langchain_anthropic import ChatAnthropic

        if isinstance(model, ChatAnthropic):
            return {"type": "any"}
    except ImportError:  # pragma: no cover
        pass
    # OpenAI / DeepSeek 及多数 OpenAI 兼容端点
    return "required"


# ---------- 中间件：开局强工具 + 阶段守卫 + 串行化 + 错误归一 + hook ----------

class PhaseGuardMiddleware(AgentMiddleware):
    """每个游戏会话一个实例（随 compiled agent 按 session 缓存）。"""

    def __init__(self, session: Any) -> None:
        super().__init__()
        self._session = session
        # 同轮多个 tool_call 默认可能被 ToolNode 并发执行；用锁强制串行
        self._tool_lock = threading.Lock()

    def wrap_model_call(self, request, handler):
        # 开局回合（IDLE）模型偶尔只输出文本而不调 start_game：
        # 在真正开局成功前强制其调用工具；工具白名单仍由 wrap_tool_call 兜底。
        if self._session.status == GameStatus.IDLE and request.tool_choice is None:
            request = request.override(
                tool_choice=_force_any_tool_choice(request.model)
            )
        return handler(request)


    def wrap_tool_call(self, request, handler):
        session = self._session
        tool_call = request.tool_call
        name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id")

        # 显式绑定会话：即使工具在线程池中执行，contextvar 也必定指向本局
        token = set_current_session(session)
        try:
            with self._tool_lock:
                if not is_tool_allowed(session, name):
                    output = tool_denied_message(session, name)
                    response = ToolMessage(content=output, tool_call_id=tool_call_id)
                else:
                    try:
                        response = handler(request)
                    except TypeError as e:
                        output = f"[ERROR] 参数错误：{e}"
                        response = ToolMessage(
                            content=output, tool_call_id=tool_call_id, status="error"
                        )
                    except Exception as e:  # 工具异常归一为 [ERROR] 回灌模型
                        output = f"[ERROR] {e}"
                        response = ToolMessage(
                            content=output, tool_call_id=tool_call_id, status="error"
                        )

                result_text = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                trigger_hooks("PostToolUse", {"tool": name, "result": result_text})
                return response
        finally:
            reset_current_session(token)


# ---------- compiled agent 缓存（按 会话 + system_prompt） ----------
#
# Web：开局使用含"最近人物排除区块"的 prompt，对局中的 ask/guess 使用不含区块
# 的 prompt，因此同一 session 至多缓存两个 compiled agent；CLI 全程一个。

_agent_cache: dict[tuple[str, int], Any] = {}
_agent_cache_lock = threading.Lock()


def _get_agent(session: Any, system_prompt: str, max_tokens: int):
    key = (session.session_id or str(id(session)), hash(system_prompt))
    with _agent_cache_lock:
        agent = _agent_cache.get(key)
        if agent is None:
            model = get_chat_model(max_tokens=max_tokens)
            agent = create_agent(
                model=model,
                tools=list(LANGCHAIN_TOOLS),
                system_prompt=system_prompt,
                middleware=[PhaseGuardMiddleware(session)],
            )
            _agent_cache[key] = agent
        return agent


def _last_ai_text(messages: list) -> Optional[str]:
    """取最后一条带文本的 AIMessage 的文本（等价旧版 last_text）。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = msg.text
            if text:
                return text
    return None


def run_agent_loop(
    session: Any,
    messages: list,
    *,
    system_prompt: str,
    max_iterations: int = 8,
    max_tokens: int = 2000,
) -> Optional[str]:
    token = set_current_session(session)
    try:
        agent = _get_agent(session, system_prompt, max_tokens)
        # 每次"模型→工具"往返在图中计 2 步，留 2 步余量
        recursion_limit = max(4, max_iterations * 2 + 2)

        try:
            result = agent.invoke(
                {"messages": list(messages)},
                config={"recursion_limit": recursion_limit},
            )
        except GraphRecursionError:
            # 跑满 max_iterations 仍未完成合法工具调用（死循环/屡次调错）：
            # 不回写图内消息、绝不用上一轮旧文本冒充本轮回复；
            # 返回 None，由编排层给出语义正确的中性重试提示（422，不扣次数）
            return None
        except Exception as e:  # LangChain/LangGraph/网络异常统一归一
            raise UpstreamLLMError(str(e)) from e

        # 用图状态中的完整消息历史原位替换传入列表（累积语义与旧版一致）
        updated = result["messages"]
        messages.clear()
        messages.extend(updated)
        return _last_ai_text(updated)
    finally:
        reset_current_session(token)
