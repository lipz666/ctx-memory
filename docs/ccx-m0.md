# ccx M0：拆分、影子代理与 LOCA-bench 接入（2026-10-03）

设计见 [ccx-design.md](ccx-design.md)。M0 不改变任何请求，只做测量。

## 1. 做了什么

- **workspace**：根包 ctx 拆成库（`src/lib.rs`）和二进制；新增 `crates/ccx`，它依赖 ctx 库。`cargo build --release` 会同时生成 `ctx` 和 `ccx`。
- **`ccx serve`**：影子代理。请求原样转发给上游，每个请求在 `CCX_HOME/steps.jsonl` 里记一行：各部分的估算 token（系统提示、工具定义、用户、助手、工具调用、工具结果、推理）、消息数、最大的单个工具结果，以及上游报告的 usage。**不记录消息内容和请求头**，所以 API key 不会落盘。
- **`ccx report [--tag T]`**：汇总每个请求的输入大小（p50 / p95 / 最大值），以及静态部分和动态部分的占比。各部分的估算值会按上游报告的输入 token 等比例校准。
- **LOCA-bench**：克隆到 `bench/external/LOCA-bench`（版本 8b6fac4，目录已加入 gitignore），使用本地 `.venv`（Python 3.12）。filesystem 和 memory 两个 MCP 服务器首次运行时由 npx 拉取，不做全局安装。
- **工具脚本**：`bench/tools/loca_estimate.py`（token 预估）、`bench/tools/loca_subset.py`（每个任务只取前 N 个 seed）。

## 2. 静态前缀实测（零 API 开销）

测量方法：LOCA 指向 ccx，ccx 指向一个本地假模型。假模型对每个请求都直接调用 `claim_done`，所以每条轨迹只发一个请求。测量对象是 8K 配置的全部 75 条轨迹（15 个任务 × 5 个 seed）。

| 指标 | 值（字符数 / 4 估算） |
|---|---|
| 首个请求的大小 | 平均 5,035；最小 2,507；最大 11,991 |
| 其中工具定义 | 占 96%；不同任务分别为 2,458 / 3,377 / 3,601 / 10,579 / 11,723 |
| 系统提示 | 0（LOCA 不发系统消息） |
| 任务说明（第一条用户消息） | 49 – 354 |

结论：LOCA 的静态前缀平均约 5K，最大约 12K。ccx 的“动态 4K”预算是另外单独计算的。

## 3. token 预估

估算公式：一条轨迹的输入 ≈ 调用次数 ×（前缀 +（最终上下文 − 前缀）/ 2），即假设上下文从前缀线性增长到最终大小。

- 调用次数取论文表 5 的工具调用数加 1。Gemini 会把多个工具调用并到一次请求里，所以这个值偏高。
- 最终上下文取论文表 4 中 Gemini-3-Flash 的数据。LOCA 的计法是单次调用 total_tokens 的最大值，已经包含前缀。
- 输出按每次调用 1,500 token 估算（含思考）。

运行 `python bench/tools/loca_estimate.py steps.jsonl` 得到：

| 长度 | 调用 | 最终上下文 | 每条轨迹输入 | 75 条输入 | 75 条输出 |
|---|---|---|---|---|---|
| 8K | 21.9 | 19.8K | 0.27M | 20.4M | 2.5M |
| 16K | 23.8 | 23.8K | 0.34M | 25.7M | 2.7M |
| 32K | 29.9 | 35.3K | 0.60M | 45.3M | 3.4M |
| 64K | 29.4 | 46.2K | 0.75M | 56.5M | 3.3M |
| 96K | 34.2 | 73.7K | 1.35M | 101.0M | 3.8M |
| 128K | 39.1 | 101.4K | 2.08M | 156.1M | 4.4M |
| 256K | 34.7 | 141.7K | 2.55M | 191.0M | 3.9M |
| **全部** | | | | **596M** | **24M** |

不同运行方案（单臂，即不开 ccx 的基线）：

| 方案 | 范围 | 输入 | 输出 |
|---|---|---|---|
| A 试点 | 8K，每任务 1 个 seed，共 15 条 | **4.1M** | 0.5M |
| B 核心基线 | 8K / 32K / 128K，每任务 2 个 seed，共 90 条 | **88.7M** | 4.1M |
| C 三个长度全量 | 8K / 32K / 128K，每任务 5 个 seed，共 225 条 | 221.7M | 10.2M |
| D 全部 | 7 个长度 × 75 条 | 596M | 24M |

不确定性：分词器不同（这里按字符数 / 4 估算，实际按 Gemini 分词计）、模型不同（论文用的是 3 Flash，我们用的是 3.8 flash-high）、重试次数，都会影响结果。误差大约 ±50%。如果网关支持隐式缓存，静态前缀可以命中缓存，实际费用会更低。

M1 以后，ccx 那一臂每次调用约为“前缀 + 4K 动态”，每条轨迹大约 0.2–0.4M，基本不随长度增长，比基线便宜得多。

## 4. 运行方法（需要批准后才能执行）

```bash
# 1) 影子代理（CCX_HOME 按 run 分开）
CCX_HOME=bench/results/loca/ccx target/release/ccx serve \
  --upstream https://vps.lpzproxy.xyz --port 7789 --tag loca-pilot-8k

# 2) 子集配置（方案 A）
python3 bench/tools/loca_subset.py \
  bench/external/LOCA-bench/task-configs/final_8k_set_config.json /tmp/loca_8k_s1.json --seeds 1

# 3) LOCA（key 只通过环境变量传入；ccx 把 Authorization 原样转发给上游，自己不保存）
cd bench/external/LOCA-bench && source .venv/bin/activate
LOCA_OPENAI_API_KEY="$CTX_GW_KEY" LOCA_OPENAI_BASE_URL=http://127.0.0.1:7789/v1 \
  loca run -c /tmp/loca_8k_s1.json -m gemini-3.8-flash-high --max-workers 4 \
  -o ../../results/loca/pilot-8k

# 4) 汇总
CCX_HOME=bench/results/loca/ccx target/release/ccx report --tag loca-pilot-8k
```

- `--max-workers 4`：每个 worker 会启动 5 到 7 个 MCP 进程，本机内存 16 GB，LOCA 默认的 20 个 worker 太多。
- 已检查过：LOCA 的输出目录和日志里都不会写入 API key。

## 5. 试点 A 结果（8K，每个任务 1 个 seed；2026-10-03）

> **注意：这里的成功率无效。** 后来检查轨迹发现，8 次通过全部读过 LOCA 的评测代码或标准答案（主分支没有沙箱），详见 [ccx-m1.md](ccx-m1.md) 第 4 节。token 和上下文构成的数据仍然有参考价值。

一共运行了三次，都通过 ccx 影子代理，用网关上的 gemini-3.8-flash-high：

| 运行 | 内容 | 输入 token | 结束方式 |
|---|---|---|---|
| r1 | 15 个任务 | 约 11 万（17 个请求） | gzip bug：ccx 把 `accept-encoding` 转发给了上游，又删掉了响应里的 `content-encoding`，LOCA 解析不了只能一直重试。修复：不再转发 `accept-encoding` |
| r2（`loca-pilot-8k-r2`） | 15 个任务 | 6,259,200 | 达到 600 万上限后停止，完成 6 个 |
| rest（`loca-pilot-8k-rest`） | r2 没完成的 9 个任务，另起一个配置重新跑 | 10,179,367 | 达到 1,000 万上限后停止，完成 4 个 |

**合计**：输入 1,655 万 token，其中 **1,164 万命中缓存（70%）**；输出 7.4 万，另有推理 10.0 万；520 个请求，0 错误。

### 成功率

完成 10 个，**通过 8 个**；另外 5 个因为达到上限没有跑完。论文里 Gemini-3-Flash 在 8K 上是 64%，不过我们只完成了 10 个，而且没完成的都是较重的任务，所以这个数字偏乐观。

| 任务 | 结果 | 调用次数 | 输入合计 | 峰值上下文 |
|---|---|---|---|---|
| AcademicWarning | ✓ | 43 | 1.13M | 53K |
| ABTesting | ✓ | 40 | 0.86M | 41K |
| CanvasListTest | ✓ | 36 | 0.90M | 44K |
| CanvasArrangeExam | ✓ | 22 | 0.44M | 27K |
| ApplyPhDEmail | ✗ | 16 | 0.13M | 16K |
| CourseAssistant | ✗ | 10 | 0.12M | 14K |
| FilterLowSellingProducts | ✓ | 44 | 2.95M | 115K |
| MachineOperating | ✓ | 40 | 1.64M | 70K |
| ExcelMarketResearch | ✓ | 49 | 1.44M | 56K |
| SetConfCrDdl | ✓ | 38 | 0.81M | 44K |
| UpdateMaterialInventory、WoocommerceNewWelcome、WoocommerceStockAlert、NhlB2bAnalysis、PayableInvoiceChecker | 未完成 | | | |

完成的 10 条轨迹平均 **33.8 次调用、1.04M 输入**，是论文推算值（0.27M）的 **3.8 倍**。

### 上下文构成（ccx report）

| | r2 | rest |
|---|---|---|
| 单个请求的输入，p50 / p95 / 最大 | 17K / 106K / 116K | 35K / 80K / 115K |
| 动态部分，p50 / p95 | 9K / 103K | 30K / 76K |
| 静态部分（工具定义），p50 / 最大 | 4K / 15K | 3K / 16K |
| **工具结果占输入的比例** | **67.7%** | **73.1%** |
| 工具定义 / 工具调用 / 用户消息 | 22.6% / 8.8% / 0.9% | 16.9% / 9.3% / 0.7% |

### 发现

1. **上下文的大头是工具结果，约占 70%**。M1 的第一步就是给工具输出分级，这个方向在真实数据上得到了确认。
2. **单个大工具结果会拖垮整条轨迹**。在 r2 中，SetConfCrDdl 读到一个 6.4 万 token 的工具结果，之后每一步都要带着它，还没完成就用掉了 1.55M。rest 里同一个任务没有出现这么大的结果，0.81M 就完成了。可见 Agent 自己的行为方差就很大，评测时需要多个 seed 或多次重复。
3. **基线的 token 消耗是论文推算值的约 4 倍**。8K 每条轨迹约 1M，更长的档位只会更多。基线越贵，ccx 的对照越有意义；ccx 那一组的输入大约是“前缀 + 4K”乘以调用次数。
4. **网关有隐式缓存**，命中率 70%，实际费用远低于 token 数字面上的量。
5. **网关实际服务的模型并不固定**。r2 中返回 gemini-3.8-flash 181 次、-control 40 次、-n 15 次；rest 中 284 次全部是 gemini-3.8-flash。对比实验时要按 `usage.model` 分开统计，或者先弄清这些名字的含义。

### 对后续的影响

- 原来的方案 B（8K / 32K / 128K × 2 个 seed）按实测修正后大约需要 300M+ 输入 token，不按原样运行。
- 建议做法：先写 M1，再让基线和 ccx 在同一批任务上成对运行。8K 下还剩 5 个没完成的任务，补齐这 5 个任务的基线约需 5–10M。

### 安全：LOCA 的 Agent 运行在本机，没有沙箱

LOCA 的 python_execute、filesystem、terminal 等工具都直接在本机执行。试点中有 3 条轨迹的 Agent 列出了本机进程，进程命令行里带着网关 key（来自本会话的命令，以及另一个会话的 Hindsight 进程）。于是 key 被写进了轨迹文件，并且随对话一起发给了模型。轨迹文件里的 key 已经替换成 `[REDACTED]`，这些输出目录也已加入 gitignore，不会被提交。

以后正式运行前必须做到：

- key 不出现在任何进程的命令行里。环境变量对同一用户的进程同样可见（`ps -E`），所以光改成环境变量也不够；
- 最好在 LOCA 的 `loca-sandbox` 分支、容器或者单独的系统用户下运行，让 Agent 看不到本机的其他进程和文件。
