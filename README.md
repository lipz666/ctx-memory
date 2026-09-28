# ctx：本地长期记忆引擎

ctx 让编码 Agent 跨会话记住东西：项目的命令和约定、踩过的坑和修法、你的偏好。它是一个 Rust 单二进制程序，以本地代理和 MCP 服务的形式工作，数据只存在本机。架构说明见 [docs/architecture.md](docs/architecture.md)。

## 它做什么

- **自动记住。** Agent 的请求经过 ctx 代理时，ctx 按会话记录每一步新增的消息。会话空闲（默认 10 分钟）或收到 `task_end` 后，由模型从对话中提炼值得长期保留的内容，写成记忆：事实、教训、做法，以及你明确说过的偏好。相近的旧记忆会被更新，而不是重复写一条。
- **需要时想起来。** 出现新的用户消息或新的报错时，ctx 按当前项目检索记忆（本地语义模型 + 中文分词关键词），把相关的几条附在该消息后面，并在本会话后续每一步保留。规则（`type: rule`）始终放在系统提示里。没有相关记忆时，请求原样逐字节转发。
- **Agent 主动读写。** MCP 工具 `recall`、`remember`、`forget`、`expand`。主动检索（MCP、`ctx recall`、REST）按相关度返回更多条，并附带相关的原始对话片段：提炼漏掉的细节（金额、名字、助手给过的推荐）仍然找得回来。
- **你可以掌控。** 记忆是 `~/.ctx/memory/*.md` 文件（YAML 头 + 正文），由独立 git 仓库记录每次修改；可以手动编辑、在面板里审核，或用 CLI 管理。自动提炼永远不会自动启用规则；Agent 提议的规则要你批准。

项目自动识别：代理从 Agent 提示词里的工作目录（Claude Code、Codex、OpenClaw 都会写）取 git 仓库名；MCP 用启动目录；也可以用 `X-Ctx-Project` 请求头或 `/a/AGENT/p/PROJECT/v1` 路由指定。项目记忆只在该项目里出现，`global` 记忆处处可见。

## 安装与接入

```bash
cargo build --release
./target/release/ctx init
./target/release/ctx model set https://YOUR-GATEWAY/v1 YOUR-MODEL --credential-ref keychain:ctx-llm
./target/release/ctx connect claude-code
./target/release/ctx connect codex
./target/release/ctx connect openclaw
./target/release/ctx doctor --llm
./target/release/ctx open
```

`model set` 配置的模型用于会话提炼；不配置时仍可手动记忆和召回。凭证只以引用形式保存（`keychain:SERVICE` 或 `env:NAME`）。macOS 默认在 `~/.ctx` 保存状态，并通过 launchd 常驻；设置 `CTX_HOME=/some/path` 可建立隔离实例（不安装服务、不改 shell 配置）。

语义模型（EmbeddingGemma 300M，量化版）首次使用时下载到 `~/.cache/ctx/models`（约 330 MB，可用 `CTX_MODEL_DIR` 改位置），之后完全离线运行，单次查询约 50 ms。下载失败时自动退回关键词检索。

代理地址为 `http://127.0.0.1:7788/a/AGENT/v1`，请求需带 `X-Ctx-Token`（值在 `~/.ctx/token`）。手动接入其他 Agent：`ctx connect AGENT UPSTREAM_BASE_URL`。OpenClaw 用 `api: openai-completions` 类型的 provider 指向该地址；支持 OpenAI Chat Completions、Responses 和 Anthropic Messages，含 SSE。

## 常用命令

```bash
ctx remember '部署 payments 前先执行 make migrate' --type lesson --scope payments
ctx remember '回答一律用中文' --type rule
ctx recall '部署前要做什么'            # 默认用当前目录所在仓库作为项目
ctx memories
ctx edit mem_... '新的内容'
ctx forget mem_...
ctx review mem_... approve
ctx sessions                          # 会话与提炼状态
ctx extract --session KEY             # 立即提炼某个会话
ctx status
ctx open                              # 本地面板：搜索、审核、编辑、会话
```

记忆类型：`fact`（事实）、`lesson`（教训）、`skill`（做法）、`rule`（始终生效，只能由你创建或批准）、`intent`。触发器（`--trigger keyword:TEXT`、`error:TEXT`、`tool:NAME`、`file:GLOB`）可让某条记忆在确定情形下必定出现。

Python SDK 在 `sdk/python`，TypeScript SDK 在 `sdk/typescript`：`remember`、`recall`、`task_end`（带 session id 时立即提炼）、`sessions`、`feedback` 等。

## 配置（`~/.ctx/config.yaml`）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `recall.max_injected` | 3 | 每次新用户消息或报错最多注入几条 |
| `recall.min_similarity` | 0.34 | 自动注入的语义命中阈值（在召回评测集上调定） |
| `recall.search_min_similarity` | 0.25 | 主动检索的语义下限 |
| `recall.search_episodes` | 5 | 主动检索附带的原始对话片段条数（0 关闭） |
| `embedding.enabled` / `model` | true / embeddinggemma-300m-q | 关闭后只用关键词检索 |
| `extraction.enabled` / `idle_minutes` | true / 10 | 会话空闲多久后提炼 |
| `extraction.daily_llm_calls` | 100 | 引擎每日模型调用上限 |
| `mcp_tools` | default | `full` 额外列出 `flag_memory`、`set_intent` |
| `debug_capture` | false | 保存每步完整请求，供调试面板查看 |
| `experimental.*` | 全部关闭 | Gate、ActionGuard、上下文驱逐、定时维护，见架构文档 |

## 验证

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
cargo build
python3 tests/idle_overhead_smoke.py
python3 tests/proxy_smoke.py
python3 tests/adapter_smoke.py
python3 tests/memory_lifecycle_e2e.py
python3 tests/virtual_memory_journey.py
python3 tests/action_guard_smoke.py
cargo build --release && python3 tests/recall_eval.py
```

`tests/recall_eval.py` 在 32 条记忆、90 条自然提问（中英文、跨语言、长句、19 条不应命中的问题）上评测召回；结果见 [docs/memory-engine-acceptance.md](docs/memory-engine-acceptance.md)。需要真实网关的测试（`tests/extraction_live.py`、`tests/local_agents_live.py` 等）通过 `CTX_TEST_LIVE_*` 环境变量提供网关与凭证引用。
