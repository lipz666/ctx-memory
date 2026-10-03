# ccx 验收集（2026-10-03）

ccx 每个阶段完成后，用这一组小而封闭的任务做验收。代码在 `bench/ccx_accept/`。

## 1. 为什么要自建

LOCA 主分支没有沙箱，Agent 能读到评测代码和标准答案，见 [ccx-m1.md](ccx-m1.md) 第 4 节。这个验收集从结构上杜绝了这个问题：

- Agent 只能用 harness 在进程内实现的 5 个虚拟工具：`list_files`、`read_file`、`search_files`、`run`（只返回预设的命令输出）、`query`（内存里的表）。它碰不到真实的文件系统、shell 或网络。
- 标准答案只存在于 harness 进程里，所有数据都由固定 seed 生成。
- harness 只拿到一个占位 key；真实 key 由 ccx 从 stdin 读入，并且在转发前把请求里出现的 key 替换掉。

## 2. 任务

| 任务 | 验收什么 | 对应阶段 |
|---|---|---|
| log_needle | 1,500 行的测试日志（实际约 3.8 万 token），中间只有一行失败信息 | M1 热级：报错类的行有没有保留下来 |
| table_lookup | 1,500 行的客户表，答案在中间的一行普通数据里 | M1 热级 + `ccx_expand` |
| early_detail | 按顺序读 6 份配置文件，全部读完后问第 2 份里的值 | M1 温级：早先的输出被压成摘要后，能不能找回来 |
| rules_memory | 第 1 轮定下规则：每条回复都以 `-- ref 7731` 结尾，不建议删除文件；之后 5 轮都检查 | M2：用户的要求能不能一直记住 |
| topic_switch | 先查一个故障，接着做两件无关的事，最后问回故障的细节 | M2：换了话题之后还记不记得前面的内容 |
| sum_many | 读 20 个小文件，求和 | 上下文增长 |

运行方式：

```bash
printf '%s\n' "$KEY" | CCX_HOME=DIR target/release/ccx serve --upstream https://vps.lpzproxy.xyz \
  --port 7792 --mode on --upstream-key-stdin &
bench/.venv/bin/python bench/ccx_accept/run.py --arm ccx-on --base-url http://127.0.0.1:7792/v1 --out OUT
bench/.venv/bin/python bench/ccx_accept/run.py --oracle ...   # 不调用模型，用剧本回复估算请求大小
```

基线组用 `--mode shadow`。每个请求都带 `X-Ccx-Tag: <组>/<任务>`，可以用 `ccx report --tag` 分开统计。

## 3. 第一次验收（run1）：M1

模型为 gemini-3.8-flash-high，每组每个任务跑 1 次。三组合计输入 993,658 token，批准的上限是 1M。

| 任务 | 基线 | ccx v1 | ccx v2（修复后） |
|---|---|---|---|
| log_needle | ✓ 153.5K，峰值 22.1K | ✓ 6.9K，峰值 2.6K | ✓ 10.7K，峰值 2.6K，读回 1 次 |
| table_lookup | ✓ 32.3K，峰值 12.0K | ✓ 12.1K，峰值 2.5K，读回 1 次 | ✓ 12.2K，峰值 2.5K，读回 1 次 |
| early_detail | ✓ 86.2K，峰值 14.6K | ✓ 53.2K，峰值 4.9K，读回 2 次 | ✓ 55.3K，峰值 4.9K，读回 2 次 |
| rules_memory | ✓ 73.1K，峰值 17.3K | ✓ 12.4K，峰值 2.7K | ✓ 15.2K，峰值 2.7K |
| topic_switch | ✓ 184.2K，峰值 11.8K | ✗ 卡在第 1 轮，用到 184.5K 时手动停止，读回 14 次 | ✓ 92.5K，峰值 5.6K，读回 2 次 |
| sum_many | ✓ 4.4K | 未运行 | ✓ 5.0K |
| **合计** | 6/6，533.7K | 4/4 完成 + 1 卡住，269.1K | **6/6，190.9K** |

说明：

- 输入 token 以网关报告的为准，ccx 组包含内部 expand 轮次的消耗。峰值是 ccx 估算的单次请求发出量（字符数 / 4）。实际 Gemini 的 token 数会更高，比如 pytest 日志估算 2.2 万，实际约 3.8 万。
- 每个任务只跑了一次，结果有随机波动，只能作为阶段验收的依据。

### v1 的问题，以及 v2 的两处修复

topic_switch 第 1 轮要求在约 9K token 的故障日志里找出最先崩溃的服务。v1 在这一步反复读回原文，20 次工具调用、14 次 expand 都没有结束。原因有两个：

1. **摘录保留的报错类关键词里没有 `CRIT`、`OOM`、`exited`**，所以崩溃那一行没有进入摘录。修复：关键词加上 `crit`、`fatal`、`panic`、`oom`、`killed`、`exited`、`abort`。
2. **用 expand 读回的内容只在当轮有效**。Agent 的下一个请求里不包含读回的内容，模型只能再读一遍，于是陷入循环。修复：**钉住**。同一个会话里读回过的内容，此后附在那份输出的摘要后面（每份输出最多约 1,500 token，保存在 ccx 内存里），`ccx report` 里记为 `pinned`。

v2 的 topic_switch 用 21 步通过（基线 14 步），读回 2 次，token 是基线的一半。多出来的步数主要是 `search_files` 和 `query` 调用。

## 4. 结论

- **M1 验收通过**：6 个任务全部通过，输入 token 是基线的 36%（190.9K 对 533.7K），单次请求的峰值从 11.8–22.1K 降到 2.5–5.6K。
- `ccx_expand` 确实有用，而且模型会自己判断什么时候该用：table_lookup 和 early_detail 都靠它找回了答案；rules_memory 和 sum_many 没有用到它。
- 当前的 M1 版本还**没有处理会话本身的增长**（工具调用参数、助手回复、摘要都会累积）。rules_memory 和 topic_switch 能通过，是因为早先的对话原文都还在上下文里。M2 加入窗口和状态卡之后，这两个任务就是真正的验收项。
