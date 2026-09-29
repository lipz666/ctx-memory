# BEAM 实验方案（给云端执行的 AI）

本文档说明如何在云服务器上用 BEAM 评测 ctx。执行者是一个在云端运行的 AI：请先完整读完本文档、`README.md`、`docs/architecture.md` 和 `docs/benchmark-results.md`（最后几节是最新进度），再动手。

## 1. 目标

在 BEAM（ICLR 2026，最长 1,000 万 token 的对话记忆基准）上测出 ctx 的成绩，重点是 1M 和 10M 两档，并与公开成绩对比量级：

| 系统 | 100K | 500K | 1M | 10M | 来源 |
|---|---|---|---|---|---|
| Hindsight | 73.4% | 71.1% | 73.9% | 64.1% | Hindsight 官方博客（自报） |
| Mem0 | — | — | 64.1% | 48.6% | Mem0 官方博客（自报，每次检索约 6.7–7K token） |
| Honcho | 63.0% | 64.9% | 63.1% | 40.6% | 同 Hindsight 博客 |
| LIGHT（论文，Llama-4-Maverick） | 35.8% | 35.9% | 33.6% | 26.6% | BEAM 论文 |
| RAG 基线（论文） | 32.3% | 33.0% | 30.7% | 24.9% | BEAM 论文 |

公开成绩的答题/评判模型与我们不同，只能比较量级。

背景（截至 2026-09-30）：ctx 在 LongMemEval_S 未参与开发的 100 题上 93–94%；与 Mem0 OSS 同条件 20 题对比 20/20 对 18–19/20，提炼 token 约为 Mem0 的 1/3。LongMemEval_S 的历史只有约 11.5 万 token，强模型直接读全文也有约 90%，区分度不够；BEAM 10M 档才能拉开差距，也是 ctx 目前设计（原始片段逐个比对、索引全在内存）的规模上限所在。

## 2. 硬性约束

- **模型只能用 `gemini-3.8-flash-high`**（提炼、答题、评判全部），经网关 `https://vps.lpzproxy.xyz/v1`（OpenAI 兼容；也支持 Anthropic 和 Gemini 原生协议）。
- **密钥只通过环境变量 `CTX_GW_KEY` 传入**，不得写进任何文件、日志、提交或 issue。
- 不提交数据集、模型缓存、`llm-cache*.sqlite`、`questions.json` 之类的大文件（`.gitignore` 已覆盖一部分，提交前检查 `git status`）。
- 测试和实验一律用独立的 `CTX_HOME`（测评框架会自动设置）；不要在没有 `CTX_HOME` 的情况下运行 `ctx init/connect/model set`。
- 网关曾出现额度用尽（429 “Individual quota reached”）：总并发不超过 100；遇到持续 429/403 时暂停并在报告中说明，不要无限重试。
- 在 `beam-experiment` 分支上工作，完成后开 PR 合入 `main`，不要直接推 `main`。

## 3. 环境准备（Linux）

```bash
# 系统依赖：git、curl、build-essential、pkg-config、libssl-dev、sqlite3、python3.12
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
git clone https://github.com/lipz666/ctx-memory.git && cd ctx-memory
git checkout -b beam-experiment
cargo build --release && cargo test --release          # 先确认 Linux 上能编译、测试全绿（仓库只在 macOS 上验证过）
cd bench && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
export CTX_GW_KEY=...                                    # 由用户提供
export BENCH_CONCURRENCY=100 BENCH_MIN_AVAILABLE_GB=4
```

第一次启动 ctx 会从 Hugging Face 下载 EmbeddingGemma（约 316 MB，缓存在 `~/.cache/ctx/models`，可用 `CTX_MODEL_DIR` 改位置）。

冒烟检查：`python3 tests/memory_upgrades_live.py 1`（需要 `CTX_GW_KEY`，几次模型调用）。

## 4. 数据

- `Mohammadta/BEAM`：splits `100K`、`500K`、`1M`（parquet，分别 20、35、35 段对话）。
- `Mohammadta/BEAM-10M`：10 段对话（约 344 MB）。
- 共 2,000 道人工校验的问题，覆盖 10 种能力：信息提取、多跳/多会话推理、知识更新、时间推理、拒答、矛盾消解、事件排序、指令遵循、偏好遵循、总结。
- 评测代码：<https://github.com/mohammadtavakoli78/BEAM>（MIT）。数据许可 CC BY-SA 4.0。
- 下载到 `bench/data/beam/`（不要提交）。先读数据集 README 和 BEAM 仓库的评测代码，确认字段：会话/轮次边界、时间锚点、问题与评分要点（nuggets）的格式。

## 5. 需要实现：`bench/tracks/beam.py`

参考 `bench/tracks/a_longmemeval.py` 的结构（抽样、每题记录、`summary.json`、token 统计、`BENCH_RESUME` 续跑、`BENCH_ANSWER_SALT` 重复答题）。要点：

1. **每段对话一个独立的 ctx 实例**（`adapters/ctx.py` 的 `Ctx`/`CtxService`，各自的 `CTX_HOME`），对话结束后停掉进程、删除该实例目录（结果先写盘）。10M 对话约产生 3 万多个原始片段，单进程约 3–4 GB 内存，多段对话不要放进同一个进程。
2. **按会话顺序写入**：若数据自带会话边界和时间，按其切分，`observed_at` 用会话时间；若没有，按轮次切成伪会话（例如每 20 轮一个，保持原顺序），在报告里写明切分方式。每个会话调用一次 `ingest_session`（导入 + 同步提炼）。提炼失败的会话跳过并计数（与现有做法一致）。
3. **分片**：`--shard i/n` 按对话编号取模，让多台机器或多个进程各跑一部分；结果写到 `results/beam/<split>/<system>/shard-i/`，另写一个合并脚本汇总。
4. **答题**：对每道问题调用 `search_scored`（search 模式，`episodes=5`），用 `BENCH` 统一的答题提示词（LongMemEval 官方检索 + 逐步推理提示词，`a_longmemeval.py` 里的 `ANSWER_PROMPT`；如果 BEAM 仓库有官方答题提示词，优先用官方的，并在报告中写明）。
5. **两档检索预算**：主配置 `budget=2000`（与我们之前所有结果一致）；另跑一个复用同一记忆库只重新答题的 `budget=8000` 变体（与 Hindsight 的 8,192、Mem0 的约 7K 可比）。复用方式参照 `ctx-v05-reanswer`（`reuse_from`，不重新提炼）。
6. **评分**：使用 BEAM 官方的要点评分（0 / 0.5 / 1，按能力平均），评判模型为 `gemini-3.8-flash-high`。把官方评测代码复制进来时保留其 MIT 版权声明（参照 `bench/data/evaluate_qa_official.py` 的做法）。
7. **记录**：每题一行 `rows.jsonl`（能力类别、得分、带回 token、检索延迟、错误）；`summary.json` 汇总总分、各能力得分、提炼调用数和 token（ctx 的 `/api/v1/stats` 有 `engine_input_tokens`/`engine_output_tokens`）、答题评判 token、墙钟时间。

先写单元级自检（例如用 100K 的 1 段对话、3 道题跑通全流程），再上规模。

## 6. 执行顺序与预算

ctx 提炼输入约为对话 token 的 1.3 倍，实测处理速度约 13.5 万 token/分钟/10 核（瓶颈是本机 CPU 计算向量，不是网关）。

| 步骤 | 内容 | 预计提炼 token | 10 核机器耗时 | 通过条件 |
|---|---|---|---|---|
| 1 | 100K 全部 20 段 | 约 3.5M | 约 30 分钟 | 流程无错误；得分量级合理；给出各能力得分 |
| 2 | 1M 抽 10 段（种子固定，写进报告） | 约 13M | 约 2 小时 | 与 Mem0 64.1%、Hindsight 73.9% 比量级 |
| 3 | 10M 先跑 1 段 | 约 14M | 约 2 小时 | 内存峰值、检索延迟、失败会话数都在可接受范围 |
| 4 | 10M 全部 10 段 | 约 130–150M | 约 18 小时（32 核约 3–5 小时） | 最终成绩 |

每一步结束都先汇报（成绩、token、耗时、问题），再进入下一步；第 4 步开始前需要用户确认预算。

资源建议：32 vCPU / 64 GB 可同时跑 4–6 段 10M 对话；不需要 GPU。内存按“同时运行的对话数 × 4 GB”预留，`BENCH_MIN_AVAILABLE_GB` 设为 4。

## 7. 已知的坑

- **不要用挂起（SIGSTOP）暂停**：长请求会超时，整轮作废。要停就终止，然后用 `BENCH_RESUME=1` 续跑（已完成的会话跳过，导入但未提炼的只重新提炼）。
- 提高并发不一定更快：CPU 占满时，瓶颈是向量计算。
- 网关偶尔返回空内容，ctx 会重试；连续失败的会话被跳过并计入 `failed_sessions`。
- 同一题两次答题可能不同（答题模型的波动），重要结论至少答两遍（`BENCH_ANSWER_SALT`）。
- 10M 规模下原始片段逐个比对向量会变慢；如果检索 p95 超过 10 秒或内存超限，记录下来作为 ctx 的规模问题，不要为了跑通改动引擎的默认行为。需要修改引擎时，单独提交并在报告中说明。

## 8. 报告

- 在 `docs/benchmark-results.md` 末尾新增一节“BEAM”，写清：数据切分和抽样（种子）、会话切分方式、配置（`ctx --version`、预算、答题与评判模型、提示词来源）、各档总分和 10 种能力得分、两档预算的对比、token 和耗时、失败会话数、与公开成绩的量级对比和可比性说明。
- 结果文件放在 `bench/results/beam/`（`rows.jsonl`、`summary.json`；不要提交数据集和缓存）。
- 提交信息按仓库现有风格；完成后推送 `beam-experiment` 分支并开 PR，PR 描述里放总表。
