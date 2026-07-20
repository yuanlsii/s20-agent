# S20 Minimal Agent

这是一个从零实现的、可替换模型提供商的最小 Agent。它借鉴 `learn-claude-code`
S20 的核心控制流，但不复制 S20 的所有后台任务、MCP、团队和插件复杂度。

## 运行

在 PowerShell 中进入本目录：

```powershell
cd C:\Users\EDY\Documents\Codex\2026-07-20\agent\work\s20-agent
$env:DEMO_MODE="1"
python server.py
```

浏览器打开 <http://127.0.0.1:8765/>。Demo 模式不需要密钥，支持普通回答、计算器、本地文档搜索，以及新建、选择和恢复会话。

接入 Sub2API 时，只通过环境变量配置，不要把密钥写进代码或前端：

```powershell
$env:DEMO_MODE="0"
$env:SUB2API_BASE_URL="https://sub2api-yuanlsii.zeabur.app/v1"
$env:SUB2API_API_KEY="在这里粘贴你的 key"
$env:SUB2API_MODEL="你的模型名"
python server.py
```

如果服务要求不同的模型名，以 Sub2API 控制台实际可用模型为准。当前适配的是
`POST /v1/chat/completions`，请求使用 Bearer token，并支持 OpenAI 风格的 `tool_calls`。

## 测试

```powershell
python -m py_compile agent_runtime.py server.py
python -m unittest discover -s tests -v
Invoke-RestMethod http://127.0.0.1:8765/api/health
```

`tests/test_runtime.py` 覆盖基础工具、持久化和上下文配对；
`tests/test_requirements.py` 是需求验收测试，覆盖四步 Agent loop、直接回答、工具循环、
工具 schema、mock search、Sub2API 输出解析、Session 隔离、纯对话追问、上下文压缩、最大轮次、
工具异常和 trace。两份测试都使用标准库 `unittest`，不依赖真实 API key。

## 结构

- `agent_runtime.py`：会话、上下文压缩、工具注册、Agent loop、Sub2API 适配器。
- `server.py`：无第三方依赖的 HTTP API。
- `static/index.html`：最小浏览器界面；只调用本地 API，不接触 API key。
- `knowledge/`：供 `search` 和 `read_docs` 使用的本地 Markdown 知识库。
- `runtime_data/sessions/`：JSON 会话持久化目录，运行时自动创建。

## 系统设计

系统分为五层，依赖方向从上到下：

| 层 | 组件 | 职责 |
| --- | --- | --- |
| 浏览器层 | `static/index.html` | 展示消息、创建/切换 Session、调用 HTTP API |
| HTTP 层 | `server.py` | 校验请求、映射状态码、把 `/api/chat` 交给 Runtime |
| 控制层 | `AgentRuntime` | 执行有边界的 model → tool → model loop |
| 能力层 | `ModelClient`、`ToolRegistry` | 调用 Sub2API；注册 Schema 并执行本地工具 |
| 状态层 | `SessionStore`、`ContextManager` | 持久化会话；组织、压缩和召回上下文 |

一次请求的完整链路：

```text
浏览器 POST /api/chat
→ server.py 解析 session_id/message
→ AgentRuntime 加载 Session 并追加 user 消息
→ ContextManager 构造模型上下文
→ ModelClient 接收 messages + tools Schema
→ 直接回答，或 ToolRegistry 执行 tool_calls
→ tool 结果写回 Session 后继续调用模型
→ 保存最终 assistant 消息
→ HTTP 返回 answer/trace/summary
```

## HTTP API

- `GET /api/health`：运行状态。
- `GET /api/sessions`：按最近更新时间列出会话。
- `POST /api/sessions`：新建空会话。
- `GET /api/sessions/{id}`：读取会话及可见对话历史。
- `POST /api/chat`：在指定会话中运行 Agent。

## 和 S20 的对应关系

| S20 概念 | 本项目实现 |
| --- | --- |
| `agent_loop` | `AgentRuntime.run` |
| 工具 schema/handler | `ToolRegistry` |
| 历史压缩 | `ContextManager.compact` |
| 会话状态 | `SessionStore` |
| provider 调用 | `OpenAICompatibleClient` |
| 可重复测试 | `DemoClient` |

为了保持最小实现，本项目没有实现 S20 的 background、cron、MCP、task team、权限审批和
多 provider 路由。这些应该在最小 loop 被真实验证后再逐项加入。

## 实现说明

### Agent loop

`AgentRuntime.run()` 是关键入口，执行四步控制流：

```text
Step one   接收用户输入，写入当前 Session
Step two   构造 context，把消息和工具 Schema 发给 LLM
Step three 如果 LLM 返回 tool_calls，执行工具并追加 tool 结果
Step four  再次调用 LLM；返回最终文本或达到最大轮次
```

代码中的 `max_steps` 默认是 12。每个 Session 有独立锁，避免同一会话并发写入；不同 Session 可以并行、
互不污染。

### 工具注册和 Schema

默认注册三个工具：

| 工具 | 作用 |
| --- | --- |
| `calculator` | AST allow-list 限制的基础算术计算 |
| `search` | 在 `knowledge/*.md` 中执行确定性本地搜索，可作为 mock search |
| `read_docs` | 读取知识库 Markdown，并阻止路径逃逸 |

每个工具同时保存本地 handler 和发送给 LLM 的 JSON Schema：

```python
registry.register(name, description, parameters, handler)
```

LLM 只能根据 Schema 选择工具；真正的 Python handler 始终由服务端根据工具名解析执行。

### LLM 输出解析

`OpenAICompatibleClient` 把 provider 响应归一化成 `ModelResponse`，提取三类数据：

- `content`：最终答案
- `reasoning_content` / `reasoning`：可选 reasoning，仅写入 trace
- `tool_calls`：工具名、调用 id 和 JSON 参数

reasoning 是观测数据，不会被自动当成事实重新注入上下文；工具结果和最终回答才是会话状态。

### Session 和 Context

每个 Session 保存为 `runtime_data/sessions/<session_id>.json`。Session A 和 Session B 使用不同文件，
因此可以分别保存各自的追问、工具结果和状态。

`ContextManager` 构造发送给 LLM 的消息：

```text
system prompt
+ 旧消息摘要（发生压缩时）
+ 最近消息原文
```

默认保留最近 20 条消息，超过 80,000 字符时生成最多 20,000 字符的本地摘要。压缩时不会把
`assistant tool_call` 与对应的 `tool result` 拆开。

### Memory 的召回时机与放置方式

当前实现没有向量数据库、Embedding、关键词检索或跨 Session 的长期 Memory。这里的 Memory 实际由两部分组成：

```text
Session.messages  最近对话、assistant tool_calls、tool results、最终回答
Session.summary   被压缩的旧对话文本
```

持久化位置：

```text
runtime_data/sessions/<session_id>.json
```

召回时机不是“用户提到某个关键词时”才触发，而是每次调用 LLM 之前都执行。具体发生在
`AgentRuntime.run()` 每一轮调用 `model.complete()` 之前：

```python
messages = context.build_messages(session, system_prompt)
response = model.complete(messages, registry.schemas())
```

`build_messages()` 的放置顺序固定为：

```text
1. system message：系统约束
2. system message：Session.summary（存在时）
3. recent messages：最近的 user/assistant/tool 消息原文
```

工具 Schema 不放进消息文本，而是通过 Chat Completions 的 `tools` 参数单独发送。工具结果放在
`role="tool"` 消息中；provider reasoning 只进入 trace，不进入 Memory，也不会重新作为事实注入上下文。

工具调用后会立即再次召回同一 Session 的上下文，因此下一轮模型能够看到刚刚写入的 tool result。用户之后追问时，
也会重新加载同一 JSON 文件，并召回该 Session 的摘要和最近历史。

当前 Memory 边界：

- 只在同一个 Session 内有效，不跨 Session 共享。
- 摘要是本地截断和拼接，不是 LLM 语义总结。
- 没有相关性排序；摘要和最近消息会无条件加入每次模型请求。
- `Session.todos` 字段目前没有工具读写，因此不能视为已经实现的待办 Memory。
- 浏览器 `localStorage` 只记住最近选择的 Session ID，不保存对话正文。

### 异常和 Trace

工具异常被转换为结构化结果，而不是让整个进程崩溃：

```json
{"ok": false, "error": "具体错误信息"}
```

每次成功运行都会返回 `user`、`model`、`reasoning`、`tool_call`、`tool_result` 等 trace 事件，
方便调试和测试。

## 测试说明

测试文件位于：

```text
tests/test_runtime.py       # 基础工具、Session 持久化、上下文配对
tests/test_requirements.py  # 需求验收：loop、Schema、解析、隔离、压缩、异常、trace
```

运行全部检查：

```powershell
python -m py_compile agent_runtime.py server.py run_demo.py
python -m unittest discover -s tests -v
python -m pytest -q
```

测试使用 mock model 和 mock HTTP provider，不需要真实 API key，也不会访问 Sub2API。

## 流式交互

浏览器发送 `POST /api/chat/stream` 后，服务端以 Server-Sent Events 返回：

- `progress`：安全的高层处理进度，例如理解问题、判断工具、整理结果。
- `answer_delta`：最终答案的增量片段。
- `done`：本轮完整结果，结构与 `/api/chat` 一致。
- `error`：本轮异常信息。

provider 返回的 `reasoning_content` 只记录在 trace 中，不通过流式接口原样暴露；页面展示的是可读的处理进度，
避免把模型内部隐藏推理当作用户可见事实。

## Zeabur 部署

仓库根目录的 `Dockerfile` 会被 Zeabur 自动识别。容器默认以 `DEMO_MODE=1` 启动，监听平台注入的
`PORT`，因此不配置 API key 也能先验证页面和 Session 基本功能。

在 Zeabur 项目中选择 **Deploy New Service → GitHub**，部署 `yuanlsii/s20-agent`。部署成功后，
在服务的 **Domains** 中选择 **Generate Domain**，生成 `*.zeabur.app` 公网地址。

如需切换到真实 Sub2API，在服务的 **Variables** 中设置：

```text
DEMO_MODE=0
SUB2API_BASE_URL=https://sub2api-yuanlsii.zeabur.app/v1
SUB2API_API_KEY=<你的 API key>
SUB2API_MODEL=<Sub2API 支持的模型名>
```

不要把真实 API key 写入 `Dockerfile`、README 或 Git 仓库。
