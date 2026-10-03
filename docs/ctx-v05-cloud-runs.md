# ctx v0.5 云端实验：BEAM 1M 与 LongMemEval_S（2026-10-03）

目标：在 ctx v0.5 **从没见过的数据**上取得成绩，用于周报。BEAM 100K 的 400 题都参与过调参，不算留出数据；1M 的对话和 LongMemEval_S 中这 167 道题都没用过。

## 代码与环境

- 分支 `ctx-m-prototype`（本文所在提交）。引擎就是 [guide.md](guide.md) 描述的 v0.5；这个分支只在它上面加了两个默认关闭的接口参数（嵌入的 `kind:"query"`、`brief_only`）和测评脚本，不改变 v0.5 的行为。
- 构建：`cargo build --release`（测评脚本调用 `target/release/ctx`）。
- Python：`cd bench && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`。
- 数据：`.venv/bin/python tools/fetch_datasets.py beam-1M longmemeval-s`。
- 密钥只通过环境变量 `CTX_GW_KEY` 传入，不要写进任何文件、命令历史或日志。

## ctx v0.5 的配置

两个测评共用同一个引擎和写入配置，只有读取方式不同。

**模型**：提炼、主题档案、矛盾检查、简报、答题、评判，全部用 `gemini-3.8-flash-high`，经网关 `https://vps.lpzproxy.xyz/v1`（`BENCH_MODEL`、`BENCH_BASE_URL` 可改）。

**嵌入**：本地 EmbeddingGemma 300M（量化，768 维），首次使用时下载到 `~/.cache/ctx/models`；每个 ctx 实例 3 个模型副本（约 3 GB 内存）。

**写入（`~/.ctx/config.yaml` 默认值，测评脚本只改下面注明的几项）**

| 项 | 值 |
|---|---|
| 提炼提示词 | `bench/prompts/ctx-general.txt`（个人助理版：原子事实、事件日期、偏好、长期指令、会话摘要） |
| 主题档案 `extraction.dossiers` | 开（每个会话多一次调用） |
| 矛盾与更新检查 `extraction.contradictions` | 开 |
| 关闭的项 | `reflection`、`known_entities`、`turn_notes`、`narratives`、`dossier_overviews`；`recall.rerank`、`time_chains`、`entity_hops` |
| 测评脚本改的项 | `daily_llm_calls` 不限、`idle_minutes` 很大（由脚本逐会话触发提炼）、`history: false`、`global_scope: false`，每个实例换一个空闲端口 |
| 会话切分 | BEAM：每个批次按用户轮次切成不超过 24,000 字符的会话，按顺序写入并逐个提炼；LongMemEval：每个历史会话一个会话 |

**读取**

| | BEAM（`ctx-beam-v8b-reanswer-8k-brief`） | LongMemEval 主配置（`ctx-v06`） | LongMemEval 对照（`ctx-v06-reanswer-brief`） |
|---|---|---|---|
| 模式 | search + 简报 | search，不写简报 | search + 简报（与 BEAM 相同） |
| 预算 | 8,000（脚本向 ctx 请求 7,200） | 2,000（请求 1,800） | 8,000 |
| 原始片段 | 5 段 | 5 段 | 5 段 |
| 简报 | 宽取：50 条候选、12 段片段、用户原话记录（最多 64,000 字符）、最相关的 2 份主题档案；一次模型调用写成简报，最多 800 token（总结类 2,000），其余预算放检索结果 | — | 同左 |
| 答题提示词 | BEAM 官方 `answer_generation_for_rag` | LongMemEval 官方阅读提示词 | 同左 |
| 评分 | BEAM 官方逐要点评判；事件排序用 Kendall τ | LongMemEval 官方评判 | 同左 |

LongMemEval 之前的 93–94%（100 道留出题）是用较早的引擎（`f984c1f`）跑的，那时还没有主题档案、矛盾检查和简报；这次是第一次用完整的 v0.5 跑。

## 实验 1：BEAM 1M

先跑前 10 段对话（200 题）。分两步：先写入，并用 8k 不带简报答一遍；再在同一个记忆库上用 8k + 简报重新答题（主成绩）。

```bash
cd bench
export CTX_GW_KEY=...   # 不回显地输入
mkdir -p results/beam/1M
nohup sh -c '.venv/bin/python -m tracks.beam --split 1M --first 10 --systems ctx-beam-v8b --budget 8000 --workers 10 --out results/beam/1M --workdir work-1M &&
  .venv/bin/python -m tracks.beam --split 1M --first 10 --systems ctx-beam-v8b-reanswer-8k-brief --budget 8000 --workers 10 --out results/beam/1M --workdir work-1M' > results/beam/1M/run.log 2>&1 &
```

- 输出：`results/beam/1M/ctx-beam-v8b/`（8k 不带简报）、`results/beam/1M/ctx-beam-v8b-reanswer-8k-brief/`（主成绩）。
- 估算：每段 1M 对话约 170 个会话，逐个提炼约 2–3 小时；10 段并行约 3 小时。提炼约 2,500–3,000 万 token，答题（含简报）约 1,000 万。
- 中断后加 `BENCH_RESUME=1` 用同一条命令续跑（已提炼的会话跳过）。
- 对照：100K 上 ctx v0.5 是 67.19%（400 题）；Hindsight 自报 1M 为 73.9%、Honcho 61.8%，但模型和评判都不同，不能直接比。
- 已知风险：简报里的"用户原话记录"上限 64,000 字符。100K 时一段对话的用户消息约 4.4 万字符，能放下；1M 时要按相关度取舍，总结、排序、计数类题可能下降。这正是 1M 要检验的。

## 实验 2：LongMemEval_S（167 道从没用过的题）

500 题中有 333 道在之前的开发和测试里用过，剩下 167 道，共 7,976 个会话。

```bash
cd bench
.venv/bin/python tools/longmemeval_questions.py --out results/longmemeval-unused167
nohup sh -c '.venv/bin/python -m tracks.a_longmemeval --systems ctx-v06 --budget 2000 --out results/longmemeval-unused167 --workdir work-lme &&
  .venv/bin/python -m tracks.a_longmemeval --systems ctx-v06-reanswer-brief --budget 8000 --out results/longmemeval-unused167 --workdir work-lme' > results/longmemeval-unused167/run.log 2>&1 &
```

- 输出：`results/longmemeval-unused167/ctx-v06/`（主成绩）和 `ctx-v06-reanswer-brief/`（对照）。
- 题型分布和全集不同（时间推理 50、跨会话 47、知识更新 24、单会话-用户 20、单会话-助手 16、单会话-偏好 5、应拒答 5），偏向难题。报告时同时给出原始准确率，和按全集 500 题的题型比例加权后的准确率。
- 估算：每个会话的提炼约 4–7 千 token（带主题档案），合计约 3,500–5,500 万 token；答题很少。
- 想和公开成绩直接对齐（多数系统报的是全部 500 题）：把选题命令换成 `--all`，成本约为 3 倍；但其中 333 道参与过开发，要单独说明。

## 运行注意

- 内存：每个 ctx 实例约 3 GB（嵌入模型）。两个实验不要在同一台 16 GB 机器上同时跑；32 GB 以上可以并行。
- 长任务用 `nohup`；需要监控时看 `run.log` 和输出目录下的 `memory.log`。
- 模型调用有本地缓存（`results/llm-cache.sqlite`）；同一提示词重跑不花 token。想换随机种子重答，设 `BENCH_ANSWER_SALT=-r2`。
- 提交：只提交每个输出目录的 `rows.jsonl` 和 `summary.json`；`work-*` 记忆库、`questions.json`、缓存都不提交。
- 交接时一并给出：提交哈希、网关计量的 token、墙钟时间、失败会话数（`summary.json` 的 `usage.failed_sessions`）。
