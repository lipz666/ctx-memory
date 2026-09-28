# v0.4 alpha 验收记录

日期：2026-09-26。依据《上下文引擎技术蓝图 v1.0》第 13 节。状态：**功能路径可试用；M0、v0.1、v0.2–v0.4 均未通过退出标准**。本记录将可复测的实现结果与需要真实数据的证明分开列示；证据优先的调整见 [ADR 0004](adr/0004-evidence-first-reset.md) 与 [证据计划](evidence-plan.md)。

## 前置里程碑的未通过项

| 里程碑 | 蓝图退出标准 | 当前状态 |
| --- | --- | --- |
| M0 | 真实脱敏轨迹上的 H1 触发器命中率与精确率基线。 | **未通过**。只有虚拟回放，缺真实轨迹和人工复核标注。 |
| v0.1 · G1 | 至少两个 coding agent 一键任务级接入，≤ 60 秒上手。 | **未通过**。OpenClaw 一次真实回合通过；Hermes 仅 MCP 通过，模型回合超时。 |
| v0.1 · 成本与成功 | 每成功任务成本下降 ≥ 20%，成功率不降。 | **未通过**。缺相同任务 A/B 和经核实的实际响应模型价格；面板的基线只是估算。 |
| v0.1 · G4 | 真实负载热路径 P95 < 50 ms。 | **未通过正式验收**。本地模拟上游已通过；真实网关并发 4、ctx 仅 8 个样本的上游前耗时 P95 为 1.777 ms，样本不足。 |
| v0.1 · 稳定性及协议 | 协议一致性测试全过，连续两周内部试用无阻塞事故。 | **未通过**。已有模拟协议回归，缺录制真实流量的全矩阵及两周试用周期。 |

## 已实现并本地验证

| 路径 | 当前结果 | 验证方式 |
| --- | --- | --- |
| v0.2 自动记忆 | 强信号入持久队列，批量 Encoder，来源校验、脱敏、精确去重、同题待审、触发器宽度检查、显式审核；来自工具输出的候选不得直接变成 rule。 | `tests/automation_smoke.py`，`cargo test`。Encoder 默认关闭；模拟模型覆盖成功与审核路径。 |
| v0.3 按需召回 | 决策点 Gate 有 300 ms 超时及降级；显式反馈与漏召回上报可修复窄触发器；调试器记录原始与编译后请求；ActionGuard 可选非流式 recheck。 | `tests/action_guard_smoke.py`、`tests/proxy_smoke.py`、`cargo test`。流式工具调用维持 advisory。 |
| v0.4 巩固 | 价值分与 tier 升降、精确重复项取代、低效触发器停用、三条独立证据形成待审技能/事实、过时记忆提示核验、巩固批次整批回滚且拒绝覆盖后续人工编辑。 | `cargo test` 中的巩固与回滚用例，`tests/automation_smoke.py`。自动巩固默认关闭。 |
| 虚拟记忆任务链 | 13 条人工记忆和 9 个标注步骤验证跨项目、过期、待审、修复、巩固、回滚。 | `tests/virtual_memory_journey.py`，结果和局限见 [虚拟记忆端到端评测](virtual-memory-evaluation.md)。 |
| 真实网关 Encoder 虚拟事件 | 用户纠正事件经真实模型生成待审记忆；审核后能回放召回。 | `tests/gateway_encoder_live.py`；网关返回的围栏 JSON 与触发器字段偏差已通过解析和提示修复。此项不代表真实任务效果。 |
| v0.4 本地 Gate 分类器 | 本地哈希词袋分类器的训练、留出集精确率门槛和推理 P95 门槛已实现；仅在有至少 2,000 条效用标签且评测通过时启用。 | `cargo test`；当前没有足够样本，因此未产生可上线的模型。 |
| 基础代理与接入 | 三协议代理、SSE、MCP 六工具、Python/TypeScript SDK、OpenClaw/Hermes 隔离接入。 | `tests/proxy_smoke.py`、`tests/adapter_smoke.py`；真实 Agent 联调结果见下。 |
| 热路径 | 100 次串行、单进程、本地模拟上游：直连 P95 0.44 ms，代理 P95 1.28 ms，差值 0.83 ms。 | `tests/latency_bench.py`；不代表并发或真实网关下的 G4。 |

## 尚未满足的蓝图范围与退出标准

| 项目 | 状态与所需证据 |
| --- | --- |
| M0 / H1 | 缺少脱敏、标注“需要记忆”的历史 Agent 轨迹。`ctx eval replay` 已输出命中、精确率、弃权、注入 token 与 P95，但样例只能验证评测机制，无法给出基线。 |
| v0.2 质量与经济性 | 尚无 200 条生成记忆的人工抽检、被标错率、真实触发精确率和经核实的网关价格；不能声称准确率 ≥ 80%、被标错率 < 5% 或引擎花费占节省额 < 30%。 |
| v0.3 / H2 / G3 | 尚无同一任务下 pre-inference 与 recheck 的 A/B，缺少成功率和重复错误标签；无法证明召回用上率 ≥ 40%、Gate 超时率 < 5% 或 30 天重复犯错率下降。 |
| v0.4 / G5 / H3 | 尚无连续 30 天任务成功率曲线与漏召回时间序列；只能证明批次回滚机制，不能声称自我进化的效果已验证。 |
| 巩固全套任务 | 当前是确定性精确去重和保守的待审综合；语义聚类、证据冲突裁决、自动状态矛盾核验、独立的 L0/L1 快照重建与全量触发器收窄尚未实现。 |
| 自动漏召回检测 | 当前需要显式反馈/漏召回上报，尚未在失败后自动反向搜索并由模型判断反事实效用。 |
| 本地分类器上线 | 所需至少 2,000 条带效用标签的 Gate 样本尚未积累。 |
| 大规模与发布 | 10 万条记忆、1,000 万事件下的性能、并发与崩溃注入、安全审计、签名和 SBOM 尚未验证。 |

## 本机联调

已执行 `cargo build --release` 和 `ctx init`；`~/.ctx/bin/ctx` 与 release 构建的 SHA-256 一致，本机 launchd 服务运行。`ctx status` 显示 OpenClaw、Hermes 两条路由且存储正常；`ctx doctor --llm` 返回 OK，网关报告的实际模型名为 `gemini-3.8-flash`，与请求名 `gemini-3.8-flash-high` 不同，实际路由需由网关侧确认。模型凭证从 Keychain 引用读取，不写入仓库。

`tests/local_agents_live.py` 通过：OpenClaw 完成真实 Agent 回合并发现六个 MCP 工具；Hermes 发现六个 MCP 工具。该脚本不执行 Hermes 模型回合；此前原生提示词请求在指定网关超时，故 Hermes 任务级接入仍未验收。两者均使用临时 Agent 配置目录，没有改动用户现有 Agent 配置。

2026-09-26 起本机内部试用已开启 Encoder，并设每日 20 次引擎 LLM 调用上限；Gate 与定时巩固保持关闭，ActionGuard 为 advisory。开启时生产记忆数为 0、待处理任务数为 0。当前网关回复为数秒量级，超过 Gate 的 300 ms 限制，因此 Gate 暂未纳入 dogfood。自动路径的真实工作质量仍未验证。

## 复测

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build
python3 tests/automation_smoke.py
python3 tests/action_guard_smoke.py
python3 tests/proxy_smoke.py
python3 tests/adapter_smoke.py
python3 tests/latency_bench.py
python3 tests/virtual_memory_journey.py
PATH="$HOME/.npm-global/bin:$HOME/.hermes/venvs/hermes-dev/bin:$PATH" python3 tests/local_agents_live.py
```

默认自动开关关闭。若需试用自动路径，可按 README 明确启用，并先审查待审记忆、调用预算和回滚批次。
