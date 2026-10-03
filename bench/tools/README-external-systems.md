# BEAM：外部记忆系统的部署与运行

在 BEAM 100K 上，把 Mem0 OSS、Cognee、Hindsight 和 ctx-m 放在同样的条件下比较。结果和分析见 [docs/ctx-m-design.md](../../docs/ctx-m-design.md) 第 10 节。

## 统一条件

- 写入、答题、评判都用同一个模型（`BENCH_MODEL`，默认 gemini-3.8-flash-high），走同一个网关（`BENCH_BASE_URL`），密钥只从环境变量 `CTX_GW_KEY` 读取，不写进任何文件。
- 嵌入统一用 ctx 自带的本地 EmbeddingGemma，经 ctx 的 `/api/v1/embeddings` 提供。需要先 `cargo build --release`（`target/release/ctx`）。
- 答题和评分统一用 BEAM 官方答题提示词和逐要点评判（`bench/tracks/beam.py`）。
- 每个系统的写入流程和检索参数照搬它自己的 BEAM 配置。
- 开发集：对话 1–10；测试集：对话 11–20（只在里程碑时跑）。
- 数据：`bench/data/beam/100K.parquet`（Hugging Face `Mohammadta/BEAM` 的 100K 分片，见 docs/beam-experiment-plan.md）。

## 各系统

| 系统 | 版本 | 安装位置 | 适配器 | 测试的配置 |
|---|---|---|---|---|
| Mem0 OSS | mem0ai 2.2.1 | `bench/.venv`（`pip install mem0ai[nlp]==2.2.1 fastembed`） | `adapters/mem0_adapter.py`（`BeamMem0`） | `mem0-1k`、`mem0-8k` |
| Cognee | 1.6.2 | `bench/external/cognee-venv` | `adapters/cognee.py` + `tools/cognee_beam.py` | `cognee-1k`、`cognee-native` |
| Hindsight | 0.10.2 | `bench/external/hindsight-venv`（服务），客户端 `hindsight-client==0.10.2` 装在 `bench/.venv` | `adapters/hindsight.py` | `hindsight-1k`、`hindsight-amb` |
| ctx-m 原型 | — | `bench/ctxm` | `adapters/ctxm.py` | `ctxm-v1-render`、`ctxm-v1-render.r2` |

安装两个独立环境（`bench/external/` 不进 git）：

```bash
cd bench/external
uv venv cognee-venv --python 3.12 && VIRTUAL_ENV=$PWD/cognee-venv uv pip install "cognee==1.6.2"
uv venv hindsight-venv --python 3.12 && VIRTUAL_ENV=$PWD/hindsight-venv uv pip install "hindsight-all==0.10.2"
cd .. && VIRTUAL_ENV=$PWD/.venv uv pip install hindsight-client==0.10.2
```

### Cognee

照 Cognee 自己的 BEAM 流程（`cognee/eval_framework/beam/REPORT.md`）：

1. `tools/cognee_beam.py prepare`：按轮次预处理成 JSON 列表（每批一个文件、每轮一项），超长轮次用模型压缩。
2. `tools/cognee_ingest_one.sh <对话目录>`：`local_ingest`（add + cognify、会话记忆、全局索引），每段对话一个独立的 Cognee 根目录 `bench/beam-cognee/cognee-store/beam-<id>`。脚本会等可用内存 ≥ 3 GB 再开始。
3. 检索：`HybridRetriever(chunks_top_k=20, entities_top_k=20)`，由 `tools/cognee_beam.py serve` 按对话起一个子进程提供。

`tools/cognee.env.sh` 设置 Cognee 的模型（`custom` + 网关）和嵌入（`openai_compatible` + 本地 ctx 嵌入服务，端口和令牌来自 `bench/beam-cognee/embedder.json`）。

### Hindsight

服务单独运行，只监听本机（照 AMB 的 BEAM 配置：每段对话一个记忆库、关闭观察整合、BEAM 专用的提炼说明、检索档位 `high`）：

```bash
cd bench/external
HINDSIGHT_API_HOST=127.0.0.1 HINDSIGHT_API_PORT=8890 HINDSIGHT_API_DATABASE_URL=pg0://hindsight-bench \
HINDSIGHT_API_LLM_PROVIDER=openai HINDSIGHT_API_LLM_BASE_URL=https://vps.lpzproxy.xyz/v1 \
HINDSIGHT_API_LLM_MODEL=gemini-3.8-flash-high HINDSIGHT_API_LLM_API_KEY="$CTX_GW_KEY" HINDSIGHT_API_LLM_TIMEOUT=300 \
hindsight-venv/bin/hindsight-api
```

不要让它监听 `0.0.0.0`：它没有鉴权。

## 运行测试集

```bash
cd bench
CTX_GW_KEY=... nohup tools/run_beam_test_split.sh mem0 cognee hindsight > results/beam/100K-test/driver.log 2>&1 &
```

参数选要跑的系统（`ctxm mem0 cognee hindsight`，默认全部），一次只跑一个；结果在 `results/beam/100K-test/<系统>/`。

## 注意

- 16 GB 机器上曾经内存溢出：嵌入服务每个模型副本约 1 GB，嵌入服务只开 1–2 个副本，系统之间不要并行；Cognee 写入最多同时 2 段对话，答题 `--workers 2`。
- 后台任务有 2 小时上限的环境里，长任务用 `nohup` 启动。
- 已完成的对话会被复用（`*.done` / `ingested` 标记），同一份记忆库可以换预算重新答题；模型调用有本地缓存（`results/llm-cache.sqlite`），重跑同样的提示词不花 token。
- 实测写入成本（开发集 10 段对话）：ctx-m 输入 1.2M；Mem0 输入 3.3M；Hindsight 输入至少 6.7M（日志只记录了慢调用）；Cognee 未统计。
