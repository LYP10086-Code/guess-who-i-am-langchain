# 猜猜我是谁 — 业务流程说明书

## 一、游戏概述

"猜猜我是谁"是一款基于问答推理的互动游戏。由 AI Agent（主持人）在心里选定一个人物（如曹操），玩家通过向 Agent 提问收集线索，最终猜出该人物身份。对局共 **10 轮**，每轮 = **1 次提问 + 1 次猜测**。

## 二、角色定义

| 角色 | 职责 |
|---|---|
| Agent（主持人） | 持有秘密人物身份；回答玩家提问；验证玩家猜测；判定胜负 |
| 玩家 | 提出判断性问题、根据线索猜测人物身份 |

## 三、核心规则

1. **对局结构**：固定 **10 轮**（可通过配置调整），每轮依次进入**提问阶段**和**猜测阶段**，顺序固定"先提问、后猜测"。
2. **提问类型**：玩家提问需为可由"是/不是"明确作答的判断性问题（如"是男性吗？""是三国时期的人物吗？"）。非判断性问题由主持人用文本引导重新提问，**不计次**。
3. **回答形式**：Agent 对每个问题必须给出**肯定/明确的二元答复**——"是"或"不是"，不得含糊或拒绝。
4. **提问与猜测均可跳过**：跳过同样消耗本轮对应的机会。
5. **猜测判定**：
   - 正确 → Agent 给出肯定答复，游戏**胜利结束**。
   - 错误 → 提示"猜测错误"，若仍有剩余轮次则进入下一轮提问。
6. **结束条件**：详见第五节。

## 四、业务流程

### 4.1 主流程

**① 初始化**
- Agent 选定（或随机抽取）目标人物，对玩家保密。
- 初始化计数器：`当前轮 = 1`，总轮数 = 10（每轮 = 1 次提问 + 1 次猜测）。
- 宣布游戏开始及规则（10 轮对局、问题需"是/不是"可答等）。

**② 每轮循环**（循环条件：`当前轮 ≤ 10` 且 未胜利）
- **提问阶段**：玩家提出一个判断性问题，Agent 回答"是"或"不是"；也可跳过（消耗本轮提问机会）。
- **猜测阶段**：玩家猜测人物 或 跳过（两者都消耗本轮猜测机会）。
- 猜对 → 胜利结束；猜错/跳过 → 进入下一轮。

**③ 猜测判定**
- **猜对** → 胜利，游戏结束。
- **猜错/跳过** → 若仍有剩余轮次则进入下一轮提问；若已是末轮 → 次数用尽，失败结束并揭晓答案。

### 4.2 流程图（文字版）

```
开始
  ↓
Agent 选定人物 / 初始化（第 1 轮，提问阶段）
  ↓
[每轮循环] ←─────────────────────────────┐
  ↓                                       │
提问阶段：玩家提问 → Agent 回答 是/不是     │
（可跳过，同样消耗本轮提问机会）            │
  ↓                                       │
猜测阶段：玩家猜测 或 跳过                 │
  ↓                                       │
猜测正确？ ── 是 → 胜利结束                │
  ↓ 否                                    │
还有剩余轮次？ ── 否 → 失败结束 / 揭晓答案  │
  ↓ 是                                    │
进入下一轮提问 ────────────────────────────┘
```

## 五、结束条件与结果

| 触发条件 | 结果 | Agent 行为 |
|---|---|---|
| 玩家猜测正确 | **胜利** | 肯定答复，可附简短人物介绍 |
| 末轮（第 10 轮）猜测错误或跳过 | **失败** | 揭晓答案，鼓励再来一局 |
| 玩家主动放弃 | **中止** | 揭晓答案 |

## 六、关键状态

- `目标人物`：Agent 持有，玩家不可见。
- `当前轮次`：1 → 10。
- `已提问次数 / 已猜测次数`：每轮各消耗 1 次（跳过同样消耗）。
- `游戏状态`：进行中 / 已胜利 / 已失败 / 已中止。



## 七、可靠性与故障防御体系

后端在 LangChain 1.0 `create_agent` 循环之上叠加了一套分层防御，确保模型故障（不调工具、调错工具、参数错误、网络异常）不会卡死游戏流程或破坏状态机。

### 7.1 阶段强制工具（根治"只回文本不调工具"）

| 阶段 | 策略 | 说明 |
|---|---|---|
| 开局（IDLE） | 命名强制调用 `start_game` | 通过 `tool_choice` 锁定唯一合法工具，杜绝"只输出开场白不开局" |
| 猜测（GUESS 且玩家猜测已就位） | 命名强制调用 `judge_guess` | 根治"模型只回'是/不是'文本、不推进判定"的空转 |
| 提问（QUESTION） | 不强制 | "是否判断题"的判定本质是语义任务，且存在"非判断题文本引导、不计次"的设计内对话流；由提示词纪律 + 编排层不变量断言兜底 |

### 7.2 工具失败分级计数器（渐强纠偏 + 止损）

同一回合内，工具调用连续失败（白名单拦截 / 参数错误 / 工具异常 / 业务 `[ERROR]`）按阶段分级处理：

| 阶段 | 升级阈值（渐强纠偏） | 止损阈值（短路回合） |
|---|---|---|
| 开局（IDLE） | 第 3 次 | 第 max(4, 开局重选上限 + 1) 次 |
| 猜测（GUESS） | 第 2 次 | 第 3 次 |
| 提问（QUESTION） | 第 3 次 | 第 4 次 |

- 达到**升级阈值**：回灌给模型的错误文案附加"停止重复、改为文本引导玩家"的渐强指令。
- 达到**止损阈值**：直接短路本回合，玩家看到中性提示，本轮消息自动回滚、**不计次**。
- 任一次成功调用即清零计数；计数为回合级，每回合开始重置。

### 7.3 工具边界收口

- **`end_game` 为系统保留工具**：任何阶段调用都会被阶段白名单拒绝，模型不再持有任何不可逆工具。
- 玩家主动放弃统一走显式入口：Web 版「放弃」按钮 / CLI 中止关键词。
- 提问 / 猜测白名单只保留纯推进工具（`answer_question`、`judge_guess` / `skip_guess`）。

### 7.4 网络层与最终保险丝

- 三个模型提供商统一**显式配置**请求超时 45 秒、最多重试 2 次（可用环境变量覆盖，见 9.2），不依赖 SDK 隐式默认值。
- LangGraph `recursion_limit` 作为最外层保险丝：触发截断时回合整体失败，调用方回滚玩家消息、不扣次数，状态机零修改。
- 开局接口在 LLM 调用失败时自动清理会话资源，不产生孤儿条目；对局中会话空闲 30 分钟由后台自动回收。

## 八、交互示例（参考剧本）

**正常对局：**
```
Agent: 游戏开始！我心里想了一位中国历史人物。对局共 10 轮，每轮一次提问加一次猜测，
       问题需能用"是/不是"回答。请提问或直接猜测。

玩家: 这个人是男性吗？
Agent: 是。剩余提问次数：9。可继续提问或猜测。



玩家: 我猜是曹操。
Agent: 恭喜你，答对了！就是曹操（字孟德，东汉末年杰出的政治家、
       军事家、文学家）。游戏胜利！
```

**猜测错误：**
```
玩家: 我猜是诸葛亮。
Agent: 猜测错误，请继续提问后再尝试。剩余提问次数：8。
```

## 九、项目启动指南

项目包含两种运行形态：**Web 版（前端 + 后端）** 与 **CLI 命令行版**。Web 版需同时启动后端（FastAPI）和前端（Vite）。

### 9.1 环境要求

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| Python | 3.10 及以上（已在 3.14 验证） | FastAPI 后端 / CLI |
| Node.js | 18 及以上（含 npm） | Vue3 前端 |

### 9.2 配置 LLM（必做）

后端基于 **LangChain 1.0**（`create_agent`），通过标准工厂 `init_chat_model` 接入模型，支持 Anthropic / OpenAI 兼容端点 / DeepSeek 三类提供商。

在**项目根目录**（`guess-who-i-am/.env`，与 `backend/`、`frontend/` 平级；可直接复制 `.env.example` 改名为 `.env`）配置 `.env`，用 `LLM_PROVIDER` 选择提供商，用 `MODEL_ID` 指定模型：

```ini
# 方案 A：Anthropic 官方（默认）
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-你的真实密钥
MODEL_ID=claude-sonnet-4-6

# 方案 B：国内 Anthropic 兼容端点（如智谱 GLM / Kimi 等）
# LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=你的密钥
# MODEL_ID=对应模型名
# ANTHROPIC_BASE_URL=https://你的兼容端点地址

# 方案 C：OpenAI 官方或 OpenAI 兼容端点（DeepSeek / Kimi / 智谱等）
# LLM_PROVIDER=openai
# OPENAI_API_KEY=你的密钥
# OPENAI_BASE_URL=https://api.deepseek.com/v1
# MODEL_ID=deepseek-chat

# 方案 D：DeepSeek 官方集成（langchain-deepseek，默认关闭 thinking）
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=你的密钥
# DEEPSEEK_BASE_URL=https://api.deepseek.com
# MODEL_ID=deepseek-flash
```

> 注意：
> - 不配置 `LLM_PROVIDER` 时默认走 `anthropic`；占位符 `sk-ant-xxx` 会导致 403 PermissionDeniedError，必须替换为真实密钥。
> - 切提供商后需**重启后端 / CLI** 生效；游戏的工具调用（function calling）依赖模型侧支持，建议优先使用官方集成（方案 A / D）。
> - DeepSeek 方案默认传 `extra_body={"thinking": {"type": "disabled"}}` 关闭思考，保证二元问答与工具判定稳定。

**可选环境变量**（均有默认值，不配置也能正常运行）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GWI_LLM_TIMEOUT` | `45` | LLM 请求超时（秒），三家提供商统一生效 |
| `GWI_LLM_MAX_RETRIES` | `2` | LLM 请求失败最大重试次数 |
| `GWI_MAX_ROUNDS` | `10` | 默认游戏轮数（前端单局传参可覆盖，上限 20） |
| `GWI_PERSON_SCOPE` | `中国历史人物` | 默认人物范围 |
| `GWI_SESSION_TTL` | `1800` | 会话空闲回收时间（秒） |
| `GWI_START_MAX_RETRY` | `3` | 开局人物撞车时自动重选的上限次数 |

### 9.3 启动 Web 版（推荐）

需要开**两个终端**，分别启动后端与前端。

**终端 1 — 启动后端（工作目录必须是 `guess-who-i-am/`）：**

```powershell
cd guess-who-i-am
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

- 后端地址：http://localhost:8000
- 健康检查：http://localhost:8000/healthz（返回 `{"status":"ok"}` 即正常）
- 接口文档：http://localhost:8000/docs

**终端 2 — 启动前端：**

```powershell
cd guess-who-i-am/frontend
npm install
npm run dev
```

- 浏览器打开：http://localhost:5173
- 前端通过 Vite 代理把 `/api` 请求转发到 `http://localhost:8000`，无需额外配置。

> Windows PowerShell 提示"无法加载 npm.ps1，因为在此系统上禁止运行脚本"时，把 `npm` 换成 `npm.cmd` 即可（如 `npm.cmd install`、`npm.cmd run dev`）。

### 9.4 启动 CLI 版（可选）

在**项目根目录**直接运行（CLI 与 Web 共用同一套 LangChain 后端依赖）：

```powershell
pip install -r backend/requirements.txt
python code.py
```

- 运行后**自动开局**，无需手动输入开局指令。
- 开局失败时自动降级为手动模式，按提示输入任意内容重试即可。
- 对局中输入 `quit` / `exit` / `q` / `放弃` / `中止` / `结束` 任一关键词可中止游戏。
- 对局中遇到网络故障会提示重试，已发送的消息自动回滚、不扣次数。

### 9.5 常见问题

| 现象 | 原因与处理 |
|---|---|
| 开始游戏报 401/403（PermissionDenied/AuthenticationError） | `.env` 中是占位符密钥或密钥无效；确认 `LLM_PROVIDER` 与对应密钥变量（`ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`DEEPSEEK_API_KEY`）匹配，替换真实密钥后重启后端 |
| 报 `ModuleNotFoundError: langchain` / `langchain_deepseek` | 依赖未安装或装在了别的 Python 环境，在项目根目录重新执行 `pip install -r backend/requirements.txt`；可用 `python -c "import langchain; print(langchain.__version__)"` 自检（需为 1.x） |
| 主持人开局后一直只说开场白、不进入提问 | 多为模型工具调用不稳定：换用官方集成（`LLM_PROVIDER=anthropic` 或 `deepseek`），并确认 `MODEL_ID` 是支持 function calling 的模型 |
| 前端页面可打开但请求失败 | 后端未启动，或工作目录不是 `guess-who-i-am/`（会导致模块导入失败） |
| 端口被占用 | 后端改 `--port 8001`（需同步改 `frontend/vite.config.ts` 代理目标），或前端 `npm run dev -- --port 5174` |
| 会话提示已过期 | 会话仅存内存，空闲 30 分钟或游戏结束后自动释放，点「再玩一次」开新局即可 |
| 提示"开局失败，请重试" | 开局连续失败触发止损保护或达到递归上限，回合已整体回滚、不扣次数，直接重试即可；反复出现时检查网络与密钥 |
| 提示"主持人服务暂时不可用" | LLM 请求超时（默认 45 秒）或重试（默认 2 次）后仍失败；检查网络连通性与 API 密钥，稍后重试，可用 `GWI_LLM_TIMEOUT` / `GWI_LLM_MAX_RETRIES` 调整 |
| 连续两局选中同一个人物 | 正常情况下不会：系统把最近 10 局人物记录在 `data/recent_persons.json`，开局自动排除（含别名）；删除该文件可清空去重记忆 |

---

## 🙏 致谢
- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)
- [lca-lc-foundations](https://github.com/langchain-ai/lca-lc-foundations?utm_source=chatgpt.com)
