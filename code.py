#!/usr/bin/env python3
"""
guess-who-i-am / code.py - 猜猜我是谁 CLI（v3 薄封装）

运行（在项目根目录下）：
    python code.py

依赖：
    pip install -r backend/requirements.txt
    .env 需配置 ANTHROPIC_API_KEY / MODEL_ID（可选 ANTHROPIC_BASE_URL）

设计：
    - Agent Loop 已抽取到 backend/app/llm/agent_runner.py，CLI/Web 共用
    - 5 个工具：start_game / answer_question / judge_guess / skip_guess / end_game
    - 阶段状态机（question → guess）由 game_state.py 强制
    - hooks: UserPromptSubmit 检测放弃意图
    - 启动即自动开局（与 Web create_game 同一开局消息）；失败降级为手动输入模式
"""

import os
import sys
from pprint import pprint

# 让 `python geuss_who_i_am/code.py` 也能导入同目录模块与 backend 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import readline  # noqa: F401  保留 s01 风格
except ImportError:
    pass

from dotenv import load_dotenv

load_dotenv(override=True)

from game_state import (  # noqa: E402
    init_session, get_session, GameStatus, Phase, ResultType, GameConfig,
)
import person_history  # noqa: E402
from input_guard import looks_like_question, GUESS_INPUT_IS_QUESTION  # noqa: E402
from prompts import build_system_prompt  # noqa: E402
from hooks import trigger_hooks, ABORT_KEYWORDS  # noqa: E402
from backend.app.llm.agent_runner import run_agent_loop, UpstreamLLMError  # noqa: E402
from backend.app.llm.client import get_model  # noqa: E402
from langchain.messages import HumanMessage  # noqa: E402


# ---------- 配置 ----------

def make_config() -> GameConfig:
    cfg = GameConfig()
    cfg.person_scope = os.getenv("GWI_PERSON_SCOPE", "中国历史人物")
    if os.getenv("GWI_MAX_ROUNDS"):
        try:
            cfg.max_rounds = int(os.getenv("GWI_MAX_ROUNDS"))
        except ValueError:
            pass
    cfg.model = get_model()
    return cfg


# ---------- 状态行 ----------

def render_status_line() -> str:
    s = get_session()
    if s.status == GameStatus.IDLE:
        return "[等待开局]"
    if s.status == GameStatus.RESULT:
        rtype = s.result.type.value if s.result else "result"
        return f"[游戏结束 - {rtype}]"
    phase_zh = "提问" if s.current_phase == Phase.QUESTION else "猜测"
    return (
        f"[第 {s.current_round}/{s.config.max_rounds} 轮 · {phase_zh}阶段 · "
        f"剩余提问 {s.questions_remaining} · 剩余猜测 {s.guesses_remaining}]"
    )


# ---------- 终局处理 ----------

def handle_abort() -> bool:
    """玩家放弃：优先本地状态机直接中止（不依赖 LLM）。"""
    s = get_session()
    if s.is_terminal:
        return True
    if s.status == GameStatus.PLAYING:
        s.abort()
        print(f"游戏中止。答案是 {s.target_person}。{s.person_intro}")
    else:
        # IDLE（尚未开局）：没有可揭晓的答案，不打畸形消息
        print("已退出。本局尚未开始，没有需要揭晓的答案。")
    return True


# ---------- 入口 ----------

def main():
    cfg = make_config()
    session = init_session(cfg)#模块及实例运行的信息
    system_prompt = build_system_prompt(
        cfg.person_scope,
        cfg.max_rounds,
        exclusion_block=person_history.exclusion_block(cfg.person_scope),
    )

    print("=" * 56)
    print(" 猜猜我是谁 / Guess Who I Am")
    print("=" * 56)
    print(f" 模型: {cfg.model}    范围: {cfg.person_scope}    上限: {cfg.max_rounds} 轮")
    print(f" 输入 {' / '.join(ABORT_KEYWORDS)} 可中止游戏。")
    print("=" * 56)
    print()

    messages: list = []

    # 启动即自动开局：与 Web create_game 逐字一致的开局消息；
    # IDLE 强制 tool_choice 保证模型必须调用 start_game。
    messages.append(HumanMessage(content=(
        f"请开始游戏，人物范围：{cfg.person_scope}。共 {cfg.max_rounds} 轮，每轮先提问后猜测。"
    )))
    print("[正在开局...]")
    try:
        host_text = run_agent_loop(session, messages, system_prompt=system_prompt)
    except UpstreamLLMError as e:
        # 开局失败：回滚开局消息，降级为手动模式（用户输入任意内容重试）
        messages.clear()
        print(f"\033[33m开局失败（{str(e)[:80]}），已切换为手动模式，请输入任意内容重试。\033[0m")
    else:
        if get_session().status == GameStatus.PLAYING:
            # 已开局成功但开局话术为空文本：不清历史（上下文对后续
            # 提问/猜测仍有用），也不打印误导性提示
            if host_text:
                print(host_text)
        else:
            # 防御性降级：开局调用正常返回但状态未推进（理论上有强工具
            # 机制兜底不应发生），回滚开局消息，降级为手动模式
            messages.clear()
            print("\033[33m开局未能完成，已切换为手动模式，请输入任意内容重试。\033[0m")
    print()

    while True:
        s = get_session()
        if s.is_terminal:
            rtype = s.result.type if s.result else ResultType.ABORTED
            if rtype == ResultType.WIN:
                print(f"\n恭喜猜对！答案就是 {s.target_person}。")
            elif rtype == ResultType.EXHAUSTED:
                print(f"\n次数用完，游戏结束。答案是 {s.target_person}。")
            else:
                print(f"\n游戏已中止。答案是 {s.target_person}。")
            print("再见！")
            break

        prompt_prefix = f"\001\033[36m\002{render_status_line()} >> \001\033[0m\002"
        try:
            query = input(prompt_prefix)
        except (EOFError, KeyboardInterrupt):
            print()
            handle_abort()
            break


        text = query.strip()
        if not text:
            continue

        # UserPromptSubmit hook（含放弃意图检测）
        signals = trigger_hooks("UserPromptSubmit", query)
        if "abort" in signals:
            handle_abort()
            break
        # # QuestionCheck hook（含问题校验）
        # signals = trigger_hooks("QuestionCheck", text, s)
        




        # 冗余校验（与 Web /guesses 同一规则，见 input_guard.py）：
        # 猜测阶段误输入疑问句 → 本地确定性拦截，不调 LLM、不计次
        if s.status == GameStatus.PLAYING and s.current_phase == Phase.GUESS:
            if looks_like_question(text):
                print(f"\033[33m{GUESS_INPUT_IS_QUESTION}\033[0m")
                print()
                continue

        # 将玩家原文写入会话，供工具 handler 记录真实问题/猜测
        if s.status == GameStatus.PLAYING and s.current_phase == Phase.QUESTION:
            session.pending_question = text
            content = f"玩家提问：{text}"
        elif s.status == GameStatus.PLAYING and s.current_phase == Phase.GUESS:
            session.pending_guess = text
            content = f"玩家猜测：{text}"
        else:
            content = text

        messages.append(HumanMessage(content=content))
        try:
            host_text = run_agent_loop(session, messages, system_prompt=system_prompt)
        except UpstreamLLMError as e:
            # 网络故障：回滚刚 append 的玩家消息（与 Web orchestrator 对齐），
            # 不崩栈、不扣次数，提示后继续循环
            messages.pop()
            print(f"\033[33m主持人服务暂时不可用（{str(e)[:80]}），请稍后重试。\033[0m")
            print()
            continue
        if host_text:
            print(host_text)
        elif not get_session().is_terminal:
            # Agent 撞迭代上限（死循环/屡次调错工具）时 run_agent_loop 返回
            # None：模型侧消息未回写，撤回刚 append 的玩家消息（与 Web
            # orchestrator 对齐），给中性重试提示，不扣次数
            if messages and isinstance(messages[-1], HumanMessage):
                messages.pop()
            print("主持人刚才没能处理你的输入，请再试一次。")
        print()


if __name__ == "__main__":
    main()
