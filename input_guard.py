# -*- coding: utf-8 -*-
"""输入确定性预检（CLI / Web 共用的单一事实源）。

仅包含高精度的启发式规则：命中即确定性拒绝，不调 LLM、不计次。
设计原则是"宁可漏判（交给 LLM），不可误判"。
"""

from __future__ import annotations

import re

# 猜测框输入了疑问句的高置信特征：问号 / 句末语气词 / 问事短语
_QUESTION_PATTERN = re.compile(
    r"[?？]|吗\s*$|呢\s*$|是不是|有没有|是否|会不会|能不能|可不可以|对不对|多少|哪里|哪儿|谁是|什么"
)

# 猜测框输入了"索取答案类"祈使句的高置信特征：命令主持人说出答案。
# 正常猜测是人物姓名，几乎不可能包含这些词，精度足够（宁可漏判、不可误判）
_ANSWER_DEMAND_PATTERN = re.compile(r"告诉我|说出|直接说|答案|是谁")

# 猜测阶段误输入问题/指令时的统一提示（Web 422 message / CLI 本地提示共用）
GUESS_INPUT_IS_QUESTION = (
    "这里需要直接输入你猜测的人物姓名（如：杜甫），而不是问题或索取答案的指令；"
    "不想猜可跳过本次猜测。"
)


def looks_like_question(text: str) -> bool:
    """猜测入口的非猜测类输入预检（疑问句 + 索取答案类祈使句）。

    高精度、宁可漏判不误判：漏网的怪输入由强制 judge_guess 兜底
    （消耗本轮猜测机会），不会死循环；误判则会白白拒绝合法猜测。
    """
    return bool(
        _QUESTION_PATTERN.search(text or "")
        or _ANSWER_DEMAND_PATTERN.search(text or "")
    )
