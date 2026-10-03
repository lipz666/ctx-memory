# BEAM 100K 外部系统测试：运行记录（2026-10-03）

结果文件在 [`bench/results/beam/100K-test`](../../../results/beam/100K-test/)；完整说明在 [`docs/beam-100k-external-test-2026-10-03.md`](../../../../docs/beam-100k-external-test-2026-10-03.md)。

- `run-manifest.json`：固定实验源码、模型、数据/源码/二进制哈希、六种配置及运行时处理。
- `validation.json`：六组 200 题的独立完整性、题目/rubric 对齐与分数校验，以及原始结果哈希。
- `ingestion-validation.json`：三个系统导入覆盖证据；Cognee 50 批次、1650 轮次，Mem0/Hindsight 各 279 次会话写入。
- `*-packages.txt`：三个 Python 环境的完整冻结清单。
- `hindsight-*-evidence.txt`：当前正式运行的 BGE-small/MiniLM 启动记录与 279 次成功写入计数。
- `recovery/cognee-native/`：HTTP 524 失败记录、初次汇总、两题补跑结果、恢复说明和 199 条成功结果未变的校验证据。
- `final-status.json`：六组完成状态。

只归档结果、报告及元数据。模型、数据集输入、记忆库、虚拟环境、请求缓存与网关密钥均不在提交中。`run-manifest.json` 和补跑命令中保留的 `/workspace/...` 是实际运行路径，不是仓库内的执行路径。

在仓库根目录运行 `sha256sum -c bench/reports/beam/100K-test-2026-10-03/SHA256SUMS` 可验证本次归档文件。
