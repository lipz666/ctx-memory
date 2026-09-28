# v0.1 验收记录

日期：2026-09-25。依据《上下文引擎技术蓝图 v1.0》第 13 节。状态：**可试用候选版；退出标准尚未全部满足**。

| 项目 | 结果 | 证据或待办 |
| --- | --- | --- |
| 三种协议、普通和 SSE 代理 | 通过 | `tests/proxy_smoke.py`；真实网关的 Responses、Messages、Chat Completions 均返回 200，并产生步骤事件和用量。 |
| 工具参数与错误码保持 | 本地模拟通过 | 三种协议的工具声明、并行工具调用标记及 429 错误在代理后保持；尚未覆盖真实 Agent 的全部协议变种。 |
| 分层编译、批量驱逐和展开 | 单元测试通过 | `src/compiler.rs`、`src/engine.rs` 的测试；MCP `expand` 的 stdio 测试通过。 |
| 显式记忆、全文召回、文件/Git/SQLite | 通过基础测试 | `tests/proxy_smoke.py`、`cargo test`；无 10 万条规模测试。 |
| 本地鉴权、脱敏、加密 | 通过基础测试 | 未授权请求返回 401；事件 AES-GCM 加密；常见密钥、JWT、连接串、私钥与高熵串脱敏。安全审计仍待进行。 |
| 面板记忆与成本 | 可用 | 记忆新增、编辑、归档、步骤与 token 统计均有 API；金额统计需经核实的网关价格。 |
| Agent 接入 | 一个 Agent 真实任务通过，两个 Agent 的 MCP 通过；G1 未达标 | OpenClaw 2026.6.11 经独立配置发起完整流式 Agent 请求，返回 `OK`；其 4 个 MCP 工具可发现。Hermes Agent 0.18.0 可发现 4 个 MCP 工具，但原生模型请求超时。Claude Code 2.1.187、Codex CLI 0.155.0-alpha.9.2 包装器和模拟测试通过，真实短任务分别超时和返回 502。尚未实现两个 Agent 的可用一键接入。 |
| G4 热路径 P95 < 50 ms | 本地基准通过 | 100 次串行请求、本机模拟上游：直连 P95 0.30 ms，代理 P95 0.83 ms，差值 0.53 ms。无并发和真实网关时延波动；不能代表生产负载。 |
| G1 / M0 历史轨迹召回基线 | 待数据 | 需要脱敏且标注“需要记忆”的历史 Agent 轨迹。现有样例回放只验证脚本。 |
| 每成功任务成本降低 ≥ 20%，成功率不降 | 待 A/B 试用 | 需要任务级成功标注、对照运行和网关输入/输出/缓存 token 价格。现有原始输入 token 基线为估算，不作为收益结论。 |
| 连续两周无阻塞 Agent 事故 | 待试用周期 | 从真实内部试用开始累计，不能以一次测试替代。 |

## 复测命令

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build
python3 tests/proxy_smoke.py
python3 tests/adapter_smoke.py
python3 tests/latency_bench.py
PATH="$HOME/.npm-global/bin:$HOME/.hermes/venvs/hermes-dev/bin:$PATH" python3 tests/local_agents_live.py
```

## 解释

模型请求中使用 `gemini-3.8-flash-high`。网关在不同接口的响应中报告了不同别名，因此模型实际路由需以网关记录确认。代理不持久化请求头；上游凭证由 Keychain 引用加载。事件与记忆是两个独立真相源，FTS 索引可重建。

真实 Agent 阻塞已缩小复现：直接调用网关时，简单 Messages 请求返回 200；加入 Claude Code 的 system 内容后，即使将 system 区块展平、限制最大输出 token，也在 18–30 秒内超时。Codex CLI 请求显示本地代理地址，但真实短任务返回 502，尚未确认 502 的发起层。需网关侧核查模型路由及真实 Agent 请求兼容性后，再重跑两个 Agent 的任务级验收。

OpenClaw 的隔离测试使用约 89 KB 请求、33 个工具和 SSE；代理记录了加密步骤事件，重组的回复为 `OK`。`tests/local_agents_live.py` 用临时 Agent 目录和运行时环境变量复测，不修改用户已有的 OpenClaw/Hermes 配置。Hermes 原生请求约 72 KB，超时后缩至无工具、约 2.6 KB 的 system 提示词仍超时；直接回放仅含用户消息或通用短 system 的请求可在约 7 秒内返回 200。继续缩短 Hermes 原生 system，保留前 20 或 50 个字符时返回 200，保留前 100 个字符时超时。因而 Hermes 模型回合尚不能作为通过项，需网关侧排查提示词兼容性或更换可验证的上游模型路由。

另用 Hermes 的 `SOUL.md` 机制覆盖默认身份为简短文本、关闭两项可选指导并禁用工具后，实际 Hermes 模型回合仍在 38 秒超时。问题不只由默认身份开头一段造成。

本机已安装与 `target/release/ctx` 一致的二进制，launchd 服务运行且 `ctx doctor --llm` 通过；当前只保留 OpenClaw 和 Hermes 的代理路由。Claude Code 和 Codex 的生成包装器因真实任务未通过而解除接管，需要复测时可重新执行 `ctx connect claude-code` 或 `ctx connect codex`。

当前未实现自动 Encoder、门控复核、巩固和 ActionGuard；这些属于蓝图的后续版本。发布物签名、校验和与 SBOM 尚未交付，因此不能按公开发布标准验收。MCP 采用旧版 stdio 初始化兼容路径；新协议客户端如果不回退旧版需另行适配。

## 2026-09-27 证据更新

上文保留最初 v0.1 记录的历史表述；后续自动模块已实现为功能候选，但各版本未按退出标准验收。当前证据以 [M0 与 v0.1 证据计划](evidence-plan.md) 为准。

- 真实网关最大并发 8 的两组复测中，空记忆 30 次/臂、20 条触发记忆 20 次/臂均全部成功；ctx 热路径 P95 分别为 2.120 ms、5.177 ms，当前规模满足 G4 数值门槛。见 [G4 记录](g4-live-2026-09-27.md)。
- OpenClaw 三项仿真编码任务 A/B 严格成功率为 ctx 3/3、直连 2/3；超时的直连项验证器仍通过且用量缺失。两臂正常成功的两项中，ctx 输入 token 高 21.6%，未证明 20% 成本下降。见 [A/B 记录](ab-2026-09-27.md)。
- OpenClaw 与 Hermes 的六个 MCP 工具均已复测通过；Hermes 原生模型任务回合仍未通过。隔离 SSE 模拟可返回；同一网关下原生系统提示词反复超时，短通用系统提示词可返回。不能将其记为第二个 Agent 的任务级通过。
- 内部 Encoder 和 Gate 已开启，每日引擎 LLM 调用上限 20；有记忆的隔离 Gate 烟测在 302 ms 超时降级，主请求正常返回 200。生产实例目前无真实任务成功标签，两周稳定性尚不能开始判定。
