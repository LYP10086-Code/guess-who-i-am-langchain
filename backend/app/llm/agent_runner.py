"""Agent Loop（LangChain 1.0 / create_agent 版）：CLI / Web 共用。

run_agent_loop(session, messages, system_prompt=...)：
- 将 session 绑定到 contextvar，工具 handler 通过 get_session() 取得；
- 使用 LangChain 1.0 create_agent（底层 LangGraph）执行标准循环：
  模型 → 工具调用 → 工具结果回灌 → … → 无工具调用即结束；
- PhaseGuardMiddleware 复刻旧版三道关键语义：
  1. 阶段白名单：越权工具不执行 handler、状态零修改，回灌 [ERROR]；
  2. 同轮多个 tool_call 串行执行（对齐旧手写 for 循环，避免并发改状态机）；
  3. 工具异常归一为 [ERROR] 文本并触发 PostToolUse hook；
  4. 强工具回合：IDLE 命名强制 start_game（保证开局）、GUESS+pending_guess
     命名强制 judge_guess（根治猜测阶段"只回文本不调工具"）；阶段翻转后
     自动解除。QUESTION 阶段**不强制**——"是否判断题"的判定本质上是
     模型比正则擅长的语义任务，且该阶段存在"非判断题用文本引导、不计次"
     的设计内对话流，强制的期望代价高于收益（提示词纪律 + 编排层不变量
     断言兜底即可）。
  5. 工具失败分级计数器：同一回合内连续失败（白名单拦截/工具异常/业务
     [ERROR]）按阶段配置容忍度，达到升级阈值时在错误文案后附加"停止重复、
     改为文本"的渐强指令，达到止损阈值直接短路回合（由 run_agent_loop
     捕获后按 None 契约返回，编排层按"未推进"处理，不扣次数）。成功调用
     即清零；recursion_limit 保留为最外层保险丝。
- 无工具调用即结束，返回最后一条 assistant 文本（host_message）。
- 返回 None（仅递归上限截断时）：保证不修改 messages，调用方可据此
  安全回滚本轮追加的玩家消息（orchestrator/code.py 均依赖此契约）。
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
    Phase,
)
from hooks import trigger_hooks
from tools import (
    LANGCHAIN_TOOLS,
    is_tool_allowed,
    tool_denied_message,
    _max_start_attempts,
)

from .client import get_chat_model


class AgentLoopError(RuntimeError):
    pass


class UpstreamLLMError(AgentLoopError):
    """LLM 超时/限流/返回异常；调用方保证状态未落账。"""


class ToolGuardExhaustedError(AgentLoopError):
    """同一回合内工具调用连续失败达到止损阈值。

    由 run_agent_loop 捕获并按 None 契约返回：不回写图内消息，调用方
    （orchestrator/code.py）据此回滚玩家消息并给中性重试提示，不扣次数。
    """


def _force_named_tool_choice(model, tool_name: str) -> object:
    """各厂商"必须调用指定工具"的 tool_choice 取值。

    命名强制（而非泛泛的 required）可避免强制调工具时误触白名单里的
    破坏性工具（skip_guess / end_game）。DeepSeek 已实测支持命名强制。
    """
    try:
        from langchain_anthropic import ChatAnthropic

        if isinstance(model, ChatAnthropic):
            return {"type": "tool", "name": tool_name}
    except ImportError:  # pragma: no cover
        pass
    # OpenAI / DeepSeek 及多数 OpenAI 兼容端点
    return {"type": "function", "function": {"name": tool_name}}


# ---------- 中间件：开局强工具 + 阶段守卫 + 串行化 + 错误归一 + hook ----------

def _guard_thresholds(session: Any) -> tuple[int, int]:
    """按阶段返回（升级阈值, 止损阈值）：同一回合内连续第 N 次工具失败时
    升级错误文案，第 M 次直接短路回合。

    GUESS 已命名强制单工具，失败基本只剩参数/业务错误，容忍度最紧；
    IDLE 的止损阈值与 start_game 撞车放行上限（tools._max_start_attempts）
    对齐：保证"撞车重选"永远不会先于放行机制被止损短路。
    """
    if session.status == GameStatus.IDLE:
        return (3, max(4, _max_start_attempts() + 1))
    if session.current_phase == Phase.GUESS:
        return (2, 3)
    return (3, 4)  # QUESTION


_ESCALATION_SUFFIX = (
    "\n[系统] 工具调用已连续失败多次。请先读清楚上方错误原因："
    "可纠正则修正后重试一次；不可纠正则停止调用工具，直接用文本引导玩家。"
)


class PhaseGuardMiddleware(AgentMiddleware):
    """每个游戏会话一个实例（随 compiled agent 按 session 缓存）。"""

    def __init__(self, session: Any) -> None:
        super().__init__()
        self._session = session
        # 同轮多个 tool_call 默认可能被 ToolNode 并发执行；用锁强制串行
        self._tool_lock = threading.Lock()
        # 本回合连续工具失败计数（成功调用即清零；每回合开始由
        # run_agent_loop 经 reset_turn_guard() 重置）
        self._consecutive_failures = 0

    def reset_turn_guard(self) -> None:
        """每个 LLM 回合开始时重置失败计数（计数是回合级而非会话级）。"""
        self._consecutive_failures = 0

    def wrap_model_call(self, request, handler):
        # 开局回合（IDLE）直接命名强制 start_game：该阶段白名单只有这一个
        # 合法工具，"any/required" 泛强制反而给模型误调其他工具留了空间；
        # 命名强制只约束函数、不约束参数（person 仍由模型自选），撞车重选
        # 时同样被强制回 start_game。开局成功后阶段翻转，强制自然解除。
        if self._session.status == GameStatus.IDLE and request.tool_choice is None:
            request = request.override(
                tool_choice=_force_named_tool_choice(request.model, "start_game")
            )
        # 猜测回合（GUESS 且玩家猜测已就位）：命名强制 judge_guess，根治
        # "模型只回文本不调工具导致流程不推进"（422 空转）。judge_guess 执行后
        # 阶段自动翻转（QUESTION/RESULT），同轮后续模型调用自然解除强制。
        elif (
            self._session.status == GameStatus.PLAYING
            and self._session.current_phase == Phase.GUESS
            and (self._session.pending_guess or "").strip()
            and request.tool_choice is None
        ):
            request = request.override(
                tool_choice=_force_named_tool_choice(request.model, "judge_guess")
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
                failure = False
                if not is_tool_allowed(session, name):
                    output = tool_denied_message(session, name)
                    response = ToolMessage(
                        content=output, tool_call_id=tool_call_id, status="error"
                    )
                    failure = True
                else:
                    try:
                        response = handler(request)
                    except TypeError as e:
                        output = f"[ERROR] 参数错误：{e}"
                        response = ToolMessage(
                            content=output, tool_call_id=tool_call_id, status="error"
                        )
                        failure = True
                    except Exception as e:  # 工具异常归一为 [ERROR] 回灌模型
                        output = f"[ERROR] {e}"
                        response = ToolMessage(
                            content=output, tool_call_id=tool_call_id, status="error"
                        )
                        failure = True

                result_text = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                # handler 未抛异常但业务失败（_err 返回 "[ERROR] ..."，如
                # 开局撞车、参数不合法）同样计入连续失败
                if not failure and result_text.startswith("[ERROR]"):
                    failure = True

                if failure:
                    self._consecutive_failures += 1
                    escalate_at, abort_at = _guard_thresholds(session)
                    if self._consecutive_failures >= abort_at:
                        # 止损：短路整个回合。run_agent_loop 捕获后按 None
                        # 契约返回，编排层按"未推进"处理（不扣次数）
                        trigger_hooks(
                            "PostToolUse", {"tool": name, "result": result_text}
                        )
                        raise ToolGuardExhaustedError(
                            f"工具 {name} 同一回合内已连续失败 "
                            f"{self._consecutive_failures} 次，提前终止本回合"
                        )
                    if self._consecutive_failures >= escalate_at:
                        # 渐强纠偏：升级文案明确"可纠正则改，不可纠正则改用文本"
                        response = ToolMessage(
                            content=result_text + _ESCALATION_SUFFIX,
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                else:
                    self._consecutive_failures = 0

                final_text = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                trigger_hooks("PostToolUse", {"tool": name, "result": final_text})
                return response
        finally:
            reset_current_session(token)


# ---------- compiled agent 缓存（按 会话 + system_prompt） ----------
#
# Web：开局使用含"最近人物排除区块"的 prompt，对局中的 ask/guess 使用不含区块
# 的 prompt，因此同一 session 至多缓存两个 compiled agent；CLI 全程一个。

_agent_cache: dict[tuple[str, int], tuple[Any, PhaseGuardMiddleware]] = {}
_agent_cache_lock = threading.Lock()


def _get_agent(session: Any, system_prompt: str, max_tokens: int):
    key = (session.session_id or str(id(session)), hash(system_prompt))
    with _agent_cache_lock:
        cached = _agent_cache.get(key)
        if cached is None:
            model = get_chat_model(max_tokens=max_tokens)
            guard = PhaseGuardMiddleware(session)
            agent = create_agent(
                model=model,
                tools=list(LANGCHAIN_TOOLS),
                system_prompt=system_prompt,
                middleware=[guard],
            )
            cached = (agent, guard)
            _agent_cache[key] = cached
        return cached


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
        agent, guard = _get_agent(session, system_prompt, max_tokens)
        # 工具失败计数是回合级的：每个 LLM 回合开始时清零
        guard.reset_turn_guard()
        # 每次"模型→工具"往返在图中计 2 步，留 2 步余量
        recursion_limit = max(4, max_iterations * 2 + 2)

        try:
            result = agent.invoke(
                {"messages": list(messages)},
                config={"recursion_limit": recursion_limit},
            )
        except ToolGuardExhaustedError:
            # 工具连续失败达到止损阈值（早于 recursion_limit 的细粒度止损）：
            # 与递归上限截断同路径处理——不回写图内消息，返回 None 交由
            # 编排层给中性重试提示（422，不扣次数）
            return None
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
