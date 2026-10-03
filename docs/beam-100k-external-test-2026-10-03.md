# BEAM 100K 三个外部系统：留出测试集

状态：6/6 个配置通过完整性校验。全部完成。

范围为对话 11–20，每个配置 200 题、10 项能力各 20 题。总分为十项能力的宏平均，使用原始 BEAM 答题提示词、逐条 rubric 评分及事件排序 Kendall tau 流程。

| 配置 | 总分 | 题数 | 运行错误 | 平均注入 token（原框架估算） | 答题阶段中位检索 ms | 本配置耗时 s |
|---|---:|---:|---:|---:|---:|---:|
| mem0-1k | 48.44% | 200 | 0 | 884 | 611.3 | 3887 |
| mem0-8k | 48.37% | 200 | 0 | 903 | 767.9 | 597 |
| cognee-1k | 29.77% | 200 | 0 | 950 | 254.4 | 2568 |
| cognee-native | 46.78% | 200 | 0 | 19795 | 256.0 | 3305 |
| hindsight-1k | 38.26% | 200 | 0 | 954 | 4908.9 | 2737 |
| hindsight-amb | 48.32% | 200 | 0 | 21132 | 6343.9 | 1657 |

Cognee 的预处理及图谱导入在单独阶段完成，summary.json 的 wall_seconds 不包含这部分时间；Mem0、Hindsight 首次配置的 wall_seconds 包含导入。各系统的原始耗时口径不同。 同一系统的两种读取配置复用一次构建的记忆库。原框架复用相同答题、评分及事件对齐请求的缓存，因此后续配置的耗时和新请求数不能直接当作独立冷启动成本。Mem0 首次配置的早期耗时还包含残留进程清理；跨机器的耗时不代表严格性能复现。

## 如何解释这组结果

Mem0 的 8k 配置比 1k 低 0.07 个百分点，实际平均注入仅从 884 增到 903 个估算 token；扩大预算上限没有让该适配器提供更多长上下文。Cognee 原生配置比 1k 高 17.01 个百分点，Hindsight AMB 高 10.06 个百分点，同时平均上下文分别增至约 1.98 万和 2.11 万 token。这些结果展示各系统在两种读取配置下的表现，不能把增幅直接归因于记忆机制而忽略上下文差异。

Mem0 两档与 Hindsight AMB 的总分相差不足 0.2 个百分点。这是单轮留出测试，小差距不足以支持稳定排名。总分按十项能力的宏平均计算，不能与不同模型、答题提示词或评分流程下的官方自报成绩直接比较。

## 固定源码与执行入口

本次执行的是 [`ctx-m-prototype` 的固定提交 `4f088eeb2bb058c847193d0bb7940735ad298a88`](https://github.com/lipz666/ctx-memory/tree/4f088eeb2bb058c847193d0bb7940735ad298a88)，并非当前 `main` 的运行器。部署步骤见该提交的 [外部系统说明](https://github.com/lipz666/ctx-memory/blob/4f088eeb2bb058c847193d0bb7940735ad298a88/bench/tools/README-external-systems.md)，执行入口为 `bench/tools/run_beam_test_split.sh mem0 cognee hindsight`。模型、预算、并行度、数据和源码哈希、环境冻结清单均随结果保存；运行时缓存及连接预检处理见下文。

## 与原实验对齐的证据

- 原实验代码：`ctx-m-prototype` 分支，提交 `4f088eeb2bb058c847193d0bb7940735ad298a88`。主仓库保持原状；运行使用独立归档快照。原运行器、适配器及评分提示词哈希已核验。
- 200 题的 `(conversation, ability, index, question)` 与原留出集结果 `ctxm-v1-render.r2/rows.jsonl` 完全一致；180 道非事件排序题的逐条 rubric 列表也完全一致。
- LLM：请求 `gemini-3.8-flash-high`，统一网关 `https://vps.lpzproxy.xyz/v1`；写入、答题、评分使用该配置。网关响应的基础模型名是 `gemini-3.8-flash`，预检记录保留请求别名及 reasoning token 元数据。
- Mem0 OSS 2.2.1、Cognee 1.6.2、Hindsight 0.10.2。Mem0/Cognee 使用本地 EmbeddingGemma 300M、768 维。
- Hindsight 按用户提供的原始启动日志对齐：`BAAI/bge-small-en-v1.5`、384 维，重排为 `cross-encoder/ms-marco-MiniLM-L-6-v2`。这修正了交接文档中所有系统都使用 EmbeddingGemma 的概述。
- 读取预算：Mem0 1000/8000；Cognee 1000/原生 HybridRetriever（chunks_top_k=20、entities_top_k=20）；Hindsight 1000、不含原始块 / AMB 12288、原始块上限 8192。
- 原并行度：Mem0 5；Cognee 导入两段并行、答题 2；Hindsight 5。会话字符上限 24000，BENCH_ANSWER_SALT 为空。
- 本地 EmbeddingGemma 的原生运行库为经过官方校验的 ONNX Runtime 1.28.0。原环境的原生运行库版本和完整间接依赖冻结清单未提供，无法证明这些依赖逐位一致；本次的完整冻结清单已保存。

## 各项能力

| 能力 | mem0-1k | mem0-8k | cognee-1k | cognee-native | hindsight-1k | hindsight-amb |
|---|---:|---:|---:|---:|---:|---:|
| abstention | 80.00% | 80.00% | 85.00% | 67.50% | 75.00% | 75.00% |
| contradiction_resolution | 20.62% | 20.62% | 5.00% | 13.13% | 11.87% | 15.62% |
| event_ordering | 19.38% | 21.25% | 17.41% | 20.71% | 21.33% | 20.16% |
| information_extraction | 58.75% | 59.17% | 48.23% | 73.96% | 50.62% | 69.90% |
| instruction_following | 41.25% | 41.25% | 17.50% | 32.50% | 26.25% | 50.00% |
| knowledge_update | 60.00% | 60.00% | 30.00% | 50.00% | 37.50% | 42.50% |
| multi_session_reasoning | 47.25% | 46.00% | 24.58% | 56.46% | 41.25% | 50.29% |
| preference_following | 77.50% | 77.50% | 47.50% | 77.50% | 65.00% | 80.00% |
| summarization | 30.87% | 29.20% | 7.46% | 29.79% | 23.83% | 38.48% |
| temporal_reasoning | 48.75% | 48.75% | 15.00% | 46.25% | 30.00% | 41.25% |

Cognee 原生配置初次运行有 1 道题因网关 HTTP 524 等传输错误失败。沿用原运行器、原记忆库、提示词、模型和缓存，按原运行器的最小对话×能力范围补跑 2 题；仅替换请求失败记录，保留全部 199 条原始成功结果，修复 1 条请求失败记录。初次失败记录、初次汇总及补跑审计一并归档。该配置耗时及 reader/judge 新调用用量包含补跑。

## 运行和完整性检查

每个完成配置检查：200 条唯一记录、十段指定对话、每项能力 20 题、无题目异常、分数有限且在 [0,1]、rubric 单条评分只取 0/0.5/1、逐项与总分重新计算一致。Mem0 另检查 failed_sessions=0；Cognee 导入完成标记检查十段齐全。

Cognee 导入独立核验：50/50 批次、1650/1650 轮次，全部蒸馏状态为 completed，共生成 423 个文档；每段对话原始、解析和导入轮次数一致，未限制批次或轮次数。详见 ingestion-validation.json。

Mem0 完成 279 次会话提取、failed_sessions=0；Hindsight 的 279 次成功 retain 日志逐段与原会话数量一致（36/37/23/20/20/25/23/32/31/32），三个系统各有十段完整导入标记。Hindsight 的启动模型日志及写入计数证据纳入结果包。

初始化曾遇到模型缓存路径差异、Cognee 扩展缓存位于只读目录及其固定 30 秒连接预检超时。缓存映射已处理；Cognee 自身连接请求实测成功（默认客户端 31.21 秒），因此使用官方 COGNEE_SKIP_CONNECTION_TEST=true 选项。失败的 Cognee 尝试和未完成记忆库已移到独立目录，正式导入从新库开始。早期中断的 Hindsight 银行由原适配器的未完成重建逻辑清理。

usage 的 input/output token 是原框架计数。网关另列 reasoning_tokens，且 Cognee/Hindsight 的内部处理用量（包括导入和检索）未全部计入 summary.json，因此这些计数不是完整账单总量。密钥通过非回显输入及进程环境提供。

## 文件

- [配置、源码和数据哈希](../bench/reports/beam/100K-test-2026-10-03/run-manifest.json)
- [独立完整性校验](../bench/reports/beam/100K-test-2026-10-03/validation.json)
- [三个系统的导入覆盖校验](../bench/reports/beam/100K-test-2026-10-03/ingestion-validation.json)
- mem0-1k：[summary.json](../bench/results/beam/100K-test/mem0-1k/summary.json)、[rows.jsonl](../bench/results/beam/100K-test/mem0-1k/rows.jsonl)
- mem0-8k：[summary.json](../bench/results/beam/100K-test/mem0-8k/summary.json)、[rows.jsonl](../bench/results/beam/100K-test/mem0-8k/rows.jsonl)
- cognee-1k：[summary.json](../bench/results/beam/100K-test/cognee-1k/summary.json)、[rows.jsonl](../bench/results/beam/100K-test/cognee-1k/rows.jsonl)
- cognee-native：[summary.json](../bench/results/beam/100K-test/cognee-native/summary.json)、[rows.jsonl](../bench/results/beam/100K-test/cognee-native/rows.jsonl)
- hindsight-1k：[summary.json](../bench/results/beam/100K-test/hindsight-1k/summary.json)、[rows.jsonl](../bench/results/beam/100K-test/hindsight-1k/rows.jsonl)
- hindsight-amb：[summary.json](../bench/results/beam/100K-test/hindsight-amb/summary.json)、[rows.jsonl](../bench/results/beam/100K-test/hindsight-amb/rows.jsonl)
- 环境冻结清单：`../bench/reports/beam/100K-test-2026-10-03/bench-packages.txt`、`cognee-packages.txt`、`hindsight-packages.txt`。
- [Cognee 请求失败补跑审计](../bench/reports/beam/100K-test-2026-10-03/recovery/cognee-native/recovery.json)
- [结果与报告的 SHA-256 校验清单](../bench/reports/beam/100K-test-2026-10-03/SHA256SUMS)
