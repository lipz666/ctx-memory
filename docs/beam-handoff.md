# BEAM 对比实验交接（2026-10-03）

## 现在在哪

BEAM 100K 开发集（对话 1–10，200 题）。所有系统的条件相同：写入、答题和评判都用 gemini-3.8-flash-high，嵌入都用本地 EmbeddingGemma，答题提示词和评分都用 BEAM 官方的。完整表格见 [ctx-m-design.md](ctx-m-design.md) 第 10 节。

| 系统 | 约 1k 注入 | 各自官方配置 |
|---|---|---|
| ctx-m 原型 | **50.7 / 51.4**（v1 / r2） | — |
| ctx v0.5（8k + 简报） | — | **67.7** |
| Mem0 OSS 2.2.1 | 45.1 | 44.6（8k） |
| Hindsight 0.10.2 | 42.9 | 48.0（约 1.9 万 token） |
| Cognee 1.6.2 | 33.1 | 48.4（约 2.2 万 token） |

这些数字不是对外部系统自报成绩的复现（Hindsight 自报 73.4，Cognee 自报 79，但答题提示词、模型、评判都和我们不同）。

部署和运行方法见 [bench/tools/README-external-systems.md](../bench/tools/README-external-systems.md)。

## 要做的实验

按顺序做。命令都在 `bench/` 下运行，密钥只通过 `CTX_GW_KEY` 环境变量传入。

### 1. 测试集：三个外部系统（云端）

对话 11–20 是测试集，到现在为止还没有任何系统在上面调过参。

```bash
CTX_GW_KEY=... nohup tools/run_beam_test_split.sh mem0 cognee hindsight > results/beam/100K-test/driver.log 2>&1 &
```

- 输出在 `results/beam/100K-test/{mem0-1k,mem0-8k,cognee-1k,cognee-native,hindsight-1k,hindsight-amb}/`，每个目录有 `summary.json` 和 `rows.jsonl`。
- 本机实测：写入加答题，每个系统 1.5–3 小时，合计约 3,000 万 token。Hindsight 和 Cognee 的写入最贵。
- 内存：一次只跑一个系统（脚本已经这样安排）；机器在 32 GB 以上时，可以加大 `--workers`。

### 2. 测试集：ctx-m（本地，进行中）

`tools/run_beam_test_split.sh ctxm`，正在本机运行，结果写到 `results/beam/100K-test/ctxm-v1-render*`，跑完后提交。

### 3. ctx v0.5 在测试集上的对照

已有结果：`results/beam/100K/ctx-beam-v8b-reanswer-8k-brief-c11-20`。注意 ctx v0.5 当初是在全部 400 题上调的参，测试集对它不算留出数据。它的记忆库只在本机临时目录里，云端要用就得重建：

```bash
.venv/bin/python -m tracks.beam --split 100K --systems ctx-beam-v8b --budget 8000 --conversations 11,12,13,14,15,16,17,18,19,20 --out results/beam/100K-test --workdir WORK
.venv/bin/python -m tracks.beam --split 100K --systems ctx-beam-v8b-reanswer-8k-brief --budget 8000 --conversations 11,12,13,14,15,16,17,18,19,20 --out results/beam/100K-test --workdir WORK
```

### 4. 重复一遍，确认差距大于噪声

同一个配置换个随机种子再答一遍：设置 `BENCH_ANSWER_SALT=-r2`，答题和评判就不会命中缓存，记忆库照常复用。开发集和测试集都至少做两遍。单遍 200 题的总分波动约 ±1–2 个点，单项能力波动约 ±4 个点。

### 5. 查清 Hindsight 为什么比自报低 25 个点

外部系统的成绩要能站得住，先要排除"我们没配好"。建议把 Hindsight 的答题换成 AMB 自己的答题提示词（`vectorize-io/agent-memory-benchmark`）再答一遍，看分数能回来多少。这一步还没实现，需要在 `adapters/hindsight.py` 里加一种答题方式。

### 6. 1k 预算本身值多少分（本地，便宜）

用 ctx v0.5 的记忆库，只把简报压到 1k 以内重新答题（`--budget 1000`），约 500 万 token：

- 分数仍在 60% 以上：说明瓶颈不在 1k 预算，ctx-m 的读取应改成"多读材料，再写成 1k 简报"。
- 分数大跌：说明 1k 确实要付出代价，重心放到写入端（拆细卡片、补细节、单独做矛盾检查）。

### 7. 之后：更大规模

在 500K / 1M 上跑 ctx-m 和表现最好的外部系统，先各跑 5 段对话。BEAM 的卖点在大规模，SOTA 也是按每个规模分别比的。

## 规则

- 只在开发集（1–10）上调参；测试集只在里程碑时跑。
- 改动要在开发集上两遍都超过噪声才保留。
- 提示词里不能出现取自测评题的例子。
- 密钥不写进文件、提交或日志。
- 结果目录只提交 `rows.jsonl` 和 `summary.json`；记忆库（`bench/beam-*`）和虚拟环境不提交。
