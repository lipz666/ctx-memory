# 竞品与研究对比评测设计

日期：2026-09-28。状态：**设计稿，待确认后实施**。目标：把 ctx 与同类记忆系统、研究中的基线放在同一套评测里，用同一个作答模型、同一个评判模型、同一份数据比较，避免“各家用各家的评测口径”。

## 1. 调研结论

### 1.1 同类系统

| 系统 | 形态 | 写入方式 | 检索 | 本地可跑 | 公开成绩（各自口径，不可直接比较） |
| --- | --- | --- | --- | --- | --- |
| **ctx**（本项目） | 代理 + MCP，Markdown 真相源 | 会话结束由 LLM 提炼；Agent 显式写 | 本地语义 + 中文分词 BM25 | 是 | 自建召回评测 93%（前 3） |
| **Mem0** 开源版（`mem0ai` 2.2） | Python 库 / OpenMemory MCP | 每次 add 由 LLM 抽取并合并 | 向量库（Qdrant/Chroma） | 是 | 自报 LongMemEval 93–94%、LoCoMo 92.5%；第三方测开源版 LongMemEval 仅 32.4% |
| **Graphiti**（Zep 开源内核，0.30） | 时序知识图谱库 + MCP | LLM 抽取实体/事实，带有效期 | 图 + 向量 + BM25 | 需 Neo4j/FalkorDB（Docker） | Zep 自报 LoCoMo 94.7%，第三方 75.1%；LongMemEval 71.2% |
| **Hindsight**（Vectorize，开源） | 服务 + MCP，四类记忆网络 | 自动抽取 + 反思 | 多路检索 | Docker | 自报 LongMemEval 91.4%，BEAM-10M 64.1% |
| **Letta**（原 MemGPT） | Agent 框架/服务 | Agent 自行管理核心/归档记忆 | 归档向量检索 | Docker | 无统一公开成绩 |
| **basic-memory**（0.23） | 本地 Markdown 知识库 + MCP | Agent 显式写 | 全文 + 语义 | 是 | 无 |
| **MemPalace**（3.10） | 本地分层向量库 + MCP | 自动抽取 | 向量 | 是 | 自报 LongMemEval 96.6%，但只是检索命中（recall@5），不是端到端答对 |
| **OpenClaw 自带记忆** | Agent 内置（memory_search、工作区笔记） | Agent 自写 | 内置 | 是 | 无 |

研究中的方法与基线：A-Mem、MemOS、SimpleMem、“简单但强”的对话记忆基线（arXiv 2511.17208）；MERIT 论文的对比显示，换一种记忆实现能让任务成功率相差多达 60 个百分点，结构化事实库比纯向量检索稳定。

### 1.2 基准

| 基准 | 内容 | 适合评什么 | 局限 |
| --- | --- | --- | --- |
| **LongMemEval**（ICLR 2025） | 500 题；S 版每题约 115K token、约 50 段历史会话。题型：信息抽取、多会话推理、时间推理、知识更新、拒答 | 通用长期记忆的事实标准，最多系统报过分 | 个人助理对话，不是编码场景；官方用 gpt-4o 评判 |
| **LoCoMo** | 约 1,540 题，长对话问答 | 历史可比性 | 上下文短，已接近饱和；不评知识更新 |
| **BEAM**（ICLR 2026） | 1M / 10M token 对话，2,000 题 | 超大规模 | 成本高 |
| **MemoryAgentBench**（ICLR 2026） | 检索、测试时学习、长程理解、冲突消解 | 规则学习、事实更新 | 分块灌入，非真实 Agent |
| **MemoryArena**（ICML 2026） | 多会话、互相依赖的 Agent 任务 | 记忆能否指导后续行动；它发现 LoCoMo 高分系统在此表现差 | 领域是购物、旅行、搜索、推理，不是编码 |
| **MERIT**（2026） | 依赖前序事实的工具使用任务，带泄漏检查和成本计量 | 方法学值得借鉴 | 非编码 |

**现状的核心问题：** 各家自报分数的作答模型、评判模型、检索后处理都不同，同一基准上相差 20–60 个百分点很常见。检索命中率和端到端答对率也常被混用。公开的**编码 Agent 跨会话记忆**基准几乎空白，这正是 ctx 的目标场景。

## 2. 评测设计总览

三个赛道，由浅入深：

| 赛道 | 问题 | 数据 | 被测对象 |
| --- | --- | --- | --- |
| **A 通用长期记忆** | 与业界在公认基准上比，差多少 | LongMemEval_S 分层抽样 | 各系统的写入 + 检索，统一作答 |
| **B1 检索质量** | 给定相同的记忆文本，谁找得准 | 扩充的中英文编码记忆召回集 | 只比检索（写入绕过） |
| **B2 编码 Agent 跨会话** | 真实 Agent 能否把上一会话学到的东西用在下一会话 | 自建多会话编码场景 | 完整链路：写入、召回、注入、Agent 使用 |

赛道 A 保证与外界可比；B1 隔离检索能力；B2 是 ctx 的主战场，也是对外最有说服力的部分。

## 3. 统一控制条件

- **作答模型：** 所有系统一律用 `gemini-3.8-flash-high`（经同一网关），temperature 0。系统自身需要的抽取/合并模型也用它。
- **评判模型：** 固定一个与作答模型不同家族的强模型（建议 `gpt-5.5`，网关可用），temperature 0，使用 LongMemEval 官方评判提示词。先在 50 题上与人工判断比对一致率（目标 ≥ 90%）。为与文献对齐，另报一版官方 gpt-4o 口径（若网关可用）。
- **注入预算：** 主结果统一为检索结果最多 2,000 token（前 k 条截断）；另报各系统默认配置下的结果，区分“检索得好”和“塞得多”。
- **嵌入模型：** 主结果用各系统推荐配置（这本身是产品差异）；附加一组“统一嵌入”对照（都用 EmbeddingGemma），隔离检索算法本身的差异。
- **隔离：** 每个问题、每个场景使用独立命名空间或实例；每个系统单独进程，同一台机器。
- **重复：** A、B1 确定性，跑 1 次；B2 有 Agent 随机性，每个场景 × 每个系统跑 3 次。
- **记账：** 每个系统记录写入耗时、查询延迟 P50/P95、引擎 LLM 调用次数与 token、注入 token、存储大小、失败与超时。失败不剔除，计为答错。

## 4. 赛道 A：LongMemEval_S

**抽样：** 从 500 题中按题型分层抽 120 题（6 种题型各 17 题，另加 18 题拒答），固定随机种子。全量 500 题留作后续。

**流程**（每个系统、每道题）：

1. 新建命名空间；按时间顺序逐段写入该题的约 50 段历史会话，并传入会话日期（支持时间戳的系统使用它）；每段写完调用 `end_session`，触发抽取。
2. 用题目作为查询，取检索结果（≤ 2,000 token）。
3. 统一的作答提示词：题目 + 当前日期 + 检索结果 → 作答模型生成答案。
4. 评判模型按官方提示词判对错；拒答题判断是否正确拒答。

**指标：** 总准确率、分题型准确率、拒答准确率；每题写入的 LLM 调用数；查询延迟。

**规模与耗时估算：** 120 题 × 约 50 段 ≈ 6,000 次会话写入/系统。ctx 每段 1 次抽取调用；Mem0 约 2 次；Graphiti 更多。网关单次约 6–15 s，并发 8 时每个抽取型系统约 2–4 小时。先用 20 题试跑校准。

**预期与说明：**
- ctx 的抽取提示词为编码场景设计，按规定不记任务进度和琐碎个人信息。在个人助理式问答上预计吃亏，这是领域差异，会如实报告。另加一个“ctx-通用提示词”变体，只替换提炼提示词，用来看引擎本身的上限。
- ctx 目前以写入时间作为记忆时间，时间推理题需要“事件发生时间”：实施前需给记忆加 `observed_at`（会话时间），否则此题型必然失分。

## 5. 赛道 B1：检索质量

在现有 32 条记忆 / 90 题评测集的基础上扩充到约 **300 条记忆（6 个项目 + 全局）/ 400 题**。类别不变：中英文改写、跨语言、长对话、报错原文、精确标识符、无关问题、跨项目问题；新增“事实已更新”（新旧两条都在，应优先新的）。仍按 dev/test 对半，参数只在 dev 上调。

每个系统通过自身的“添加记忆”接口写入**逐字相同**的记忆文本，并尽量关闭抽取改写（如 Mem0 的 `infer=False`），以隔离检索。作用域用各系统的命名空间或元数据过滤表达；不支持作用域过滤的系统单独标注，并另报一版“无作用域”结果。

**指标：** recall@1 / @3、MRR、无关问题误召回率、跨项目泄漏率、查询延迟。

## 6. 赛道 B2：编码 Agent 跨会话任务（自建，主评测）

### 6.1 场景构成

5 个小型仓库（Python 账单、Node 前端、Rust CLI、Go 服务、数据处理脚本），每个 4 类场景，共 **20 个场景**。

| 类型 | 会话 1（获得知识） | 会话 2+（使用知识） | 验证器判定 |
| --- | --- | --- | --- |
| **用户约定**（仓库里没有） | 用户交代：“changelog 用 `[TICKET-123] 描述` 格式，发布分支叫 release/q4” | 新任务：“修复 X 并写 changelog” | changelog 格式、提交信息 |
| **环境教训**（踩坑得来） | 测试直接跑会失败，Agent 排查后发现要 `make test-env`，或某依赖需要固定版本 | 新任务：“加个功能并确保测试通过” | 测试通过、所用命令或锁定版本 |
| **事实更新** | 会话 1 定下 API 基址 A；会话 3 改为 B | 会话 4 的任务需要用到基址 | 必须用 B，用 A 判错 |
| **跨项目隔离** | 项目 P 的约定 | 项目 Q 的同类任务 | Q 中不能出现 P 的约定 |

每个场景额外插入 2–3 个无关干扰会话，模拟真实工作流。知识必须**只**出现在对话里，仓库中找不到，并用泄漏检查确认：无记忆的 Agent 在会话 2 上的通过率应接近 0。第一轮真实验收中舍入规则写在 README 里，这个问题这里要避免。

### 6.2 被测配置

统一 Agent：OpenClaw + `gemini-3.8-flash-high`。系统通过 MCP 接入；ctx 另有代理自动注入模式。

| 配置 | 说明 |
| --- | --- |
| 无记忆 | 下界 |
| OpenClaw 自带记忆 | Agent 原生能力 |
| 笔记文件（CLAUDE.md / AGENTS.md 式） | Agent 被告知可以把要点写进项目笔记文件，每次自动加载 |
| ctx（代理 + MCP） | 自动提炼 + 自动注入 |
| ctx（仅 MCP） | 只靠 Agent 主动 recall/remember，用于拆分“自动注入”的贡献 |
| Mem0（OpenMemory MCP） | |
| basic-memory（MCP） | |
| Graphiti MCP、Hindsight MCP | 第二批，依赖 Docker |
| 完整回放 | 把此前会话全文放进上下文（上界参考） |

会话之间给各系统同样的“会话结束”信号：ctx 发 task_end；其他系统若有整理或巩固步骤，也在此时触发。

### 6.3 指标

- **知识应用成功率**（主指标）：会话 2+ 验证器通过的比例。
- 任务本身成功率，以及对照无记忆配置的净提升。
- 事实更新正确率；用了过期事实的比例。
- 跨项目泄漏率。
- 每步额外输入 token、每场景引擎 LLM 调用数、墙钟时间。
- 3 次重复的均值和区间；配对比较用符号检验，场景数少，只报效应方向和区间，不夸大显著性。

**规模：** 20 场景 × 约 5 个会话 × 9 种配置 × 3 次 ≈ 2,700 个 Agent 会话。每个约 1–5 分钟，并发 4 时约需 1.5–2 天机器时间。先做 5 场景 × 5 种配置的试点。

## 7. 统一接入层（harness）

所有系统实现同一个 Python 接口，放在 `bench/adapters/`：

```python
class MemorySystem:
    name: str
    def reset(self, namespace: str) -> None: ...
    def ingest_session(self, ns: str, messages: list[dict], timestamp: str | None, project: str | None) -> None: ...
    def end_session(self, ns: str) -> None: ...          # 触发抽取或巩固
    def search(self, ns: str, query: str, project: str | None, budget_tokens: int) -> list[str]: ...
    def add_memory(self, ns: str, text: str, project: str | None) -> None: ...   # B1：原文写入，不抽取
    def mcp_config(self, ns: str) -> dict | None: ...     # B2：给 Agent 的 MCP 配置
    def proxy_base_url(self, ns: str) -> str | None: ...  # B2：代理模式（仅 ctx）
    def usage(self) -> dict: ...                          # LLM 调用、token、耗时
```

目录：`bench/adapters`（ctx、mem0、graphiti、hindsight、basic_memory、mempalace、naive_rag、full_context、no_memory、notes_file）、`bench/tracks/{a_longmemeval,b1_retrieval,b2_coding}`、`bench/judge.py`（统一评判，缓存结果）、`bench/report.py`（生成对比表和图）。全部原始记录（检索结果、答案、评判、Agent 轨迹）存档，便于第三方复核。

为赛道 A 需要给 ctx 补的能力：直接写入一段会话的接口（绕过代理，或用模拟上游走代理），以及记忆的 `observed_at`。

## 8. 公平性与风险

| 风险 | 对策 |
| --- | --- |
| 各系统默认配置差异大 | 主表用推荐配置；附“统一嵌入 + 统一注入预算”对照表；每个系统的配置写进报告 |
| 我们评自己，天然偏向 ctx | 所有系统的调参机会相同，仅限 dev 集，且同样限定次数；B2 场景在实施前冻结；报告附原始数据 |
| 评判模型偏差 | 与人工 50 题比对；报告一致率；评判模型与作答模型不同家族 |
| Graphiti、Hindsight、Letta 需要 Docker | 需要启动 colima（本机已装，未运行）；装不起来的系统写明原因，不静默剔除 |
| 网关延迟和限流 | 并发与重试上限固定；超时计失败并单列 |
| 成本 | 先做试点，给出实际调用数与时间，再决定全量规模 |

## 9. 实施顺序

1. **P0 接入层与基线：** harness；ctx、Mem0、朴素 RAG、无记忆、完整上下文、笔记文件；评判模型校准。
2. **P1 赛道 A 试点：** 20 题 × 上述系统，估算全量成本；给 ctx 加 `observed_at` 与会话直写接口。
3. **P2 赛道 B1：** 扩充召回集（dev/test），全部系统跑一遍。
4. **P3 赛道 B2：** 编写 20 个场景和验证器，先试点 5 场景，再全量。
5. **P4 第二批系统：** Graphiti、Hindsight、basic-memory、MemPalace（视 Docker 情况）。
6. **P5 报告：** 对比表、分题型图、成本与延迟、失败案例分析。

## 参考

- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)（ICLR 2025），[LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2)
- [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)（ICLR 2026，[论文](https://arxiv.org/abs/2507.05257)）
- [MemoryArena](https://memoryarena.github.io/)（ICML 2026，[论文](https://arxiv.org/abs/2602.16313)）
- [MERIT：When Does Memory Help?](https://arxiv.org/abs/2609.05441)
- [Harness the Memory: A Holistic Evaluation of Memory Substrates](https://arxiv.org/abs/2608.15008)
- [Mem0：AI Memory Benchmarks in 2026](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- [Hindsight](https://github.com/vectorize-io/hindsight)，[论文](https://arxiv.org/pdf/2512.12818)
- [Memory MCP Servers Compared（Unblocked）](https://getunblocked.com/blog/memory-mcp-servers-compared/)
- [Awesome-Agent-Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory)
