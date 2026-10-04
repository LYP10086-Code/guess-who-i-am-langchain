# 后端重构计划：迁移至 LangChain 1.0

> 目标：将"猜猜我是谁"后端从 **Anthropic SDK 手写 Agent Loop** 迁移到 **LangChain 1.0 系列**（`create_agent` + middleware，LangGraph 内核）。
>
> 约束：**前端与路由层完全不动**，只改后端；HTTP 契约、错误码、游戏规则零变化。
>
> 已确认决策：① CLI（code.py）保持可运行；② 模型接入顺便支持多提供商（默认 Anthropic，新增 OpenAI 兼容）；③ 采用 `create_agent` + 中间件范式；④ 保留 hooks 扩展点机制。

---

## 一、现状分析：后端是怎么跑起来的

真正与 **Anthropic SDK 强耦合**的只有 4 个点，其余（状态机、会话存储、DTO、错误码、路由、前端）都与 LLM 实现无关：

| 模块 | 现状 | 与 LLM 的耦合度 |
|---|---|---|
| `backend/app/llm/client.py` | 手写 `Anthropic` 单例，手动加载根目录 `.env`，读 `ANTHROPIC_BASE_URL` 做兼容端点 | **强耦合**，需重写 |
| `backend/app/llm/agent_runner.py` | 手写 agent loop：`client.messages.create` → 取 `tool_use` block → **顺序**派发 `TOOL_HANDLERS` → 拼 `tool_result` block 回灌，最多 8 轮；contextvar 绑定 session；异常归一为 `UpstreamLLMError` | **强耦合**，核心重写对象 |
| `tools.py` | 5 个工具用 **Anthropic 原生 schema**（`name/description/input_schema`）+ `TOOL_HANDLERS` 字典；含阶段白名单 `is_tool_allowed`、人物去重校验 | **schema 层需迁移**，handler 业务逻辑可原样保留 |
| `backend/app/services/orchestrator.py` | `entry.messages` 存的是 **Anthropic 原始 dict**（`{"role","content"}`，content 里混着 SDK block 对象）；skip/retry/abandon 以 user 消息注入 `[系统通知]`；LLM 失败时 pop 最后一条消息回滚；`asyncio.to_thread` 跑同步 loop | 消息格式需跟随迁移，编排逻辑不变 |
| `prompts.py` | 纯文本 SYSTEM prompt，工具说明写在文案里 | **零改动**（工具名保持不变） |
| `game_state.py` | 纯 Python dataclass 状态机 + contextvar/单例取会话 | **零改动** |
| `person_history.py` / `input_guard.py` | JSON 去重表 / 疑问句式预检，均为纯 Python | **零改动** |
| `hooks.py` | `UserPromptSubmit`（仅 CLI 放弃关键词）、`PostToolUse`（空实现日志点），在 runner 里触发 | 保留机制 |
| `routes/games.py`、schemas、errors、main、config、整个 frontend | HTTP 契约层 | **一行不动** |
| `code.py`（CLI） | 复用同一个 `run_agent_loop(session, messages, system_prompt=...)` | 因函数签名保持不变而零改动 |

### 当前实现必须逐条保持等价的关键行为

1. **阶段白名单**：工具派发前强校验，越权调用不执行 handler、状态零修改，回灌 `[ERROR] 当前阶段不允许…`，模型自行改道；
2. **非判断题不计次**：模型不调 `answer_question` 只输出文本时，状态不变、次数不扣，编排层据此返回 422；
3. **开局人物去重**：`start_game` 撞最近 10 人名单 → 返回 `[ERROR]` 让模型重选，最多拒绝 3 次后放行；
4. **LLM 故障无脏状态**：异常时撤回 pending 和最后一条 user 消息；
5. **工具顺序执行**：手写 for 循环按序执行同一轮的多个 tool_call；
6. 终态不变量断言、终态释放会话、30min TTL、`[系统通知]` 注入。

---

## 二、LangChain 1.0 目标架构

LangChain 1.0（2025-10-22 GA，承诺 2.0 前无破坏变更）的核心是 `create_agent`：基于 LangGraph 运行时的标准 agent loop + middleware 扩展点，配合 `langchain-anthropic` 的 `ChatAnthropic`（`anthropic_api_url` 自动回退读 `ANTHROPIC_API_URL → ANTHROPIC_BASE_URL`，**现有 `.env` 配置零改动即可用**）。

### 映射关系

| 现在的手写件 | LangChain 1.0 对应物 |
|---|---|
| `Anthropic().messages.create` | `ChatAnthropic(model, max_tokens=2000, anthropic_api_url=...)` |
| Anthropic `TOOLS` 原生 schema | `@tool` 结构化工具（类型注解/Literal 自动生成 JSON schema，中文描述原样搬进 docstring） |
| 手写 loop（create→tool_use→handler→tool_result） | `create_agent(model, tools, system_prompt, middleware=[...])`，`agent.invoke({"messages": [...]})` |
| 派发前 `is_tool_allowed` 白名单 | 每会话一个 `PhaseGuardMiddleware`（`wrap_tool_call` 短路返回 `[ERROR]` ToolMessage），工具内再留同款校验兜底 |
| `messages` 里的 Anthropic dict/block | LangChain 消息对象：`HumanMessage` / `AIMessage(tool_calls=...)` / `ToolMessage`；落内存不落盘，无历史数据迁移负担 |
| `max_iterations=8` | invoke 配置 `recursion_limit`（模型节点+工具节点计 2 步，取 18） |
| `trigger_hooks("PostToolUse")` | 由同一个 middleware 的 `wrap_tool_call` 包裹触发（hooks.py 机制保留） |
| contextvar 取 session | 保留（Python 3.9+ 线程池/任务均复制 context），`get_session()` 及 5 个 handler 内部代码不用动 |

### 消息流变化（仅内部表示，HTTP 契约不变）

```
entry.messages: [HumanMessage("玩家提问：…"),
                 AIMessage(tool_calls=[answer_question…]),
                 ToolMessage("是。剩余…", tool_call_id=…)]
skip 通知:      HumanMessage("[系统通知] 玩家跳过了本轮提问…")   ← 角色语义与现在一致
system prompt: create_agent(system_prompt=…) 每局固定一次（含排除区块），不进 messages
```

LangChain 的 Anthropic 集成会自动把 `ToolMessage` 合并转换为 Anthropic 的 `tool_result` content block，无需手工拼装。

---

## 三、文件级改动清单

### 改动 1：`backend/requirements.txt`

- 移除直接依赖 `anthropic`（由 langchain-anthropic 传递安装）；
- 新增 `langchain>=1.0`、`langchain-anthropic>=1.0`、`langchain-openai>=1.0`；
- fastapi / uvicorn / pydantic / python-dotenv 不变；
- LangChain v1 要求 Python ≥3.10，与 README 一致。

### 改动 2：重写 `backend/app/llm/client.py`（多提供商模型工厂）

- 保留根目录 `.env` 加载逻辑与 `get_model()`；
- 新增 `get_chat_model()`，用 `init_chat_model` 按环境变量选择提供商，**默认 anthropic，旧配置零改动可用**；
- 两个提供商都显式设置 `max_tokens=2000`（langchain-anthropic v1 默认值有变更，显式锁定以保持旧行为）；
- 保留现有 `ANTHROPIC_AUTH_TOKEN` 兼容端点清理逻辑。

| `.env` 变量 | anthropic（默认） | openai（OpenAI 兼容端点） |
|---|---|---|
| 提供商开关 | `LLM_PROVIDER=anthropic`（缺省即此） | `LLM_PROVIDER=openai` |
| 密钥 | `ANTHROPIC_API_KEY` | `OPENAI_API_KEY` |
| 端点 | `ANTHROPIC_BASE_URL`（可空） | `OPENAI_BASE_URL`（如 `https://api.deepseek.com/v1`） |
| 模型 | `MODEL_ID`（两边共用，如 `claude-sonnet-4-6` / `deepseek-chat`） | 同左 |

同步更新 `.env.example`：保留方案 A（Anthropic 官方）、方案 B（国内 Anthropic 兼容端点），增加注释化的方案 C（OpenAI 兼容端点）。README 默认不动。

> **实测风险**：本游戏重度依赖工具调用 + tool_result 多轮回灌，国产 OpenAI 兼容端点的 function calling 兼容度参差。Anthropic 链路作为默认并完整冒烟；OpenAI 链路在有对应密钥时实测，若兼容不佳仅在文档标注，不为此改架构。

### 改动 3：重写 `tools.py` 的 schema 层

- 5 个工具改为 `@tool` 函数，工具名与现在完全一致（prompt 文案不用动）：
  `start_game / answer_question / judge_guess / skip_guess / end_game`；
- 现有中文 `description` **逐字搬进 docstring**；
- 参数约束改为类型注解：`answer: Literal["是","不是"]`、`outcome: Literal["abort"]`、`aliases: list[str] = []`；可选参数默认值保留；
- 5 个 handler 函数体（`get_session()` 取会话、人物去重、`start_attempts` 防死循环、状态机异常转 `[ERROR]`）**原样保留**，`@tool` 只做一层薄包装；
- 导出 `LANGCHAIN_TOOLS` 替代旧 `TOOLS`；保留 `is_tool_allowed / tool_denied_message / TOOL_HANDLERS` 供 middleware 与 CLI 复用。

### 改动 4：重写 `backend/app/llm/agent_runner.py`

- **对外签名不变**：
  `run_agent_loop(session, messages, *, system_prompt, max_iterations=8, max_tokens=2000) -> Optional[str]`，
  `code.py` 因此零改动；
- 内部：`create_agent(model=get_chat_model(), tools=LANGCHAIN_TOOLS, system_prompt=..., middleware=[PhaseGuardMiddleware(session)])`，按 session 缓存 compiled agent（每局一个，CLI 全程一个）；
- 调 `agent.invoke({"messages": messages}, config={"recursion_limit": 18})`（8 轮 × 模型/工具两节点），用返回 state 的 messages 整体替换传入列表，取末尾 `AIMessage` 文本为 host_text；
- 触达迭代上限时返回最后文本，等价旧版"跑满 8 轮返回 last_text"；
- 保留 contextvar 绑定/重置、`UpstreamLLMError` 异常归一（LangChain/LangGraph 异常一并兜住）；
- `PhaseGuardMiddleware.wrap_tool_call`：
  1. 用一把本局锁把同轮多个 tool_call **串行化**（对齐旧 for 循环，消除 LangGraph ToolNode 并发执行的竞态）；
  2. 白名单不通过则短路返回 `ToolMessage(tool_denied_message(...))`，不执行 handler、状态零修改；
  3. 包裹触发 `trigger_hooks("PostToolUse", ...)`。
- **兜底**：实现时以安装版本的 `ToolCallRequest`/`Command` 实际形态为准；若中间件 API 有差异，把白名单校验放进 `@tool` 包装层（行为与现在逐字节等价）。

### 改动 5：小改 `backend/app/services/session_store.py`

- `SessionEntry.messages` 类型注释改为 `list[BaseMessage]`；
- 可选增加 `agent` 字段持有本局 compiled agent。

### 改动 6：小改 `backend/app/services/orchestrator.py`

- 仅替换消息构造（`HumanMessage(...)`）与失败回滚里对最后一条消息的类型判断（`_last_entry_is_user_prompt` 改为判断末条是 `HumanMessage` 且 content 为 str）；
- 涉及消息构造的位置：开局、提问、猜测，以及 skip/retry/abandon 的 `[系统通知]` 注入；
- **其余全部不动**：阶段守卫、预算守卫、LLM 失败消息回滚与 pending 清理、非判断题 422 判定、终态不变量断言、DTO 转换、错误码映射、终态释放/TTL、`asyncio.to_thread` 调用模式。

### 明确不改动的文件

`backend/app/api/routes/games.py`、`backend/app/schemas/`、`backend/app/errors.py`、`backend/app/main.py`、`backend/app/config.py`、`prompts.py`、`game_state.py`、`person_history.py`、`input_guard.py`、`hooks.py`、`code.py`、整个 `frontend/`。

---

## 四、执行顺序与验收

1. 建/更新虚拟环境安装新依赖，验证 langchain 1.x / langchain-anthropic 1.x / langchain-openai 1.x 无版本冲突；
2. 按序修改：`client.py` → `tools.py` → `agent_runner.py` → `session_store.py` / `orchestrator.py` → `.env.example`；
3. 后端冒烟（严格按 design.md §9）：
   - 开局人物去重；
   - 提问（yes/no + 计数 +1）；
   - 跳过提问（进 guess，提问计数 +1）；
   - 猜错（有余量自动回提问 + 提示条 + 轮次 +1 + 猜测计数 +1）；
   - 跳过猜测（进下一轮 question，猜测计数 +1）；
   - 连续猜错至 exhausted；
   - 放弃（aborted 并揭晓）；
   - GET 刷新恢复；
   - 非判断题 422 不计次；
   - 终态后 GET 404（会话已释放）；
   - 越权工具调用被 `[ERROR]` 拒绝、状态零修改；
4. CLI 冒烟：`python code.py` 完整对局一局；
5. 如有 OpenAI 兼容端点密钥，追加一轮提供商切换实测。

---

## 五、改动范围汇总

- **重写（4）**：`backend/requirements.txt`、`backend/app/llm/client.py`、`tools.py`、`backend/app/llm/agent_runner.py`
- **小改（3）**：`backend/app/services/session_store.py`、`backend/app/services/orchestrator.py`、`.env.example`
- **不新增文件**；不动前端、路由、schema、状态机。

## 六、主要风险

1. middleware 的 `wrap_tool_call` 短路返回在 1.0 小版本间的精确返回形态需以安装版本实测（兜底：白名单校验直接留在 `@tool` 函数体内）；
2. 国内 OpenAI 兼容端点经 LangChain 转换层后的工具调用兼容性需用真实密钥实测一轮；
3. 重构期间保持 `run_agent_loop` 签名稳定是 CLI 零改动的前提，修改时需同步核对 `code.py` 调用点。
