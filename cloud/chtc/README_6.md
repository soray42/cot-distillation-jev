# CHTC 第 6 批：新方法（数值节点、逐步内化），Qwen3.5-4B，10 个任务

这批是 Sora 刚写好的两类新方法，全部在 4B、GSM8K 全量数据、种子 12、学习率 3e-6 上跑，设置和第 5 批的 4B 任务完全一样（全参数微调，FSDP bf16 + cpu-adamw，不存权重）。
要和这几个任务直接比：已经跑完的只训答案 4 轮种子 12（83）、yes/no 节点的 Tree Forcing 种子 12（42），以及第 5 批的对照组（placebo）种子 12（103）。这些不重跑。

**请把这批排在第 5 批还没跑的任务前面。** DAG 里的优先级是 130–121，第 5 批是 100–46，所以两个 DAG 同时在队列里时，这批会先跑。

| 包 | 模型 | 任务 | 代码 |
|---|---|---|---|
| `cotd_chtc_6_4b_bundle.tgz` | Qwen3.5-4B | 10 个（154–163） | 当前 HEAD a91ddbe |

编号接着第 5 批（101–153）往下排，run 目录全是新的，不会覆盖 `runs/` 里已有的结果。

## 这批是什么
- **数值节点**（数据 `data/student_gsm8k_tf_ops`，4 轮）：不再问 yes/no 节点题，而是把 GSM8K 解答里的每一步原子运算变成一道三选一的数值题，例如 "What is 4 × 1.5?"。
  - GFPQV：数值题，题目文本和最终问题都在上下文里。
  - GFPQVQ：同上，前面加上这一步的子问题。
  - GFPQO：选运算式（"Which computation gives it?"，三个算式里选一个）。
  - GFPVB：只问数值题本身，不给题目文本。
  - GFPQVM：yes/no 节点和数值节点都有，各占一半权重。
- **逐步内化**（stepwise internalisation，数据 `data/student_gsm8k_tf_si`，5 轮 = 4 + 1 个阶段，每阶段 1 轮）：训练时先把解题的原子计算链写在选项前面，然后每个阶段删掉一部分步骤。最后一个阶段没有任何步骤，和测试格式一样。每个阶段开始时重置 AdamW，学习率是一条贯穿所有阶段的 warmup + cosine。
  - SIB：从答案那一端开始删。SI：从前面开始删。
  - SIYB：SIB 加上 yes/no 节点题。SISB：SIB，但链里的数值全是错的（对照）。
  - 这四个任务在每个阶段结束时自己评测一次（共 4 次），所以 env 里 `EVALAT` 是空的，这是故意的，不是漏了。
- **A2 5 轮**（157）：只训答案，训练轮数和逐步内化的 5 轮一样，作为同预算的基线。

## 关于 4B 单卡 FSDP（和第 5 批一样）
- 请像第 5 批那样，把你的 fsdp1 补丁打进 `cotd_code_data_6_4b.tgz`，每个任务用 1 张卡。
- `joblist_6_4b.txt` 第三列**已经是 1**，`jobs6_4b.dag` / `smoke6_4b.dag` 也已经按 1 卡生成好了（8 CPU、80 GB 内存、卡 ≥ 40 GB），这次**不用**再运行 `make_dags_6.sh`。env 里仍然是 `NGPU=2`，和第 5 批一样，由你的补丁处理。
- 代码从第 5 批的 8eea1dd 换成了 a91ddbe。`hpc/train_arm.sbatch` 里 NGPU / torchrun 那一段没有改，只改了两处：`arm_config` 里加了新方法的几行，训练循环里多了 `arm_config "$ARM" || true` 和 `--epochs "${EPOCHS:-${EPD:-2}}"` 这两行。`scripts/train_student.py` 和 `src/cotdistill/student.py` 也加了新功能。如果你的补丁改到这几处附近，可能需要手动合并一下。
- 如果不打补丁：把 joblist 第三列改成 2，再运行 `bash make_dags_6.sh`（2 卡：12 CPU、120 GB）。不打补丁又用 1 卡的话，任务一开始就会报 `NGPU=2 but this job sees 1 GPU(s)` 然后退出。

## 0. 解包
```bash
cd ~/cotd_chtc && tar xzf cotd_chtc_6_4b_bundle.tgz && md5sum -c MANIFEST_6_4b.md5
```
新文件有：`cotd_code_data_6_4b.tgz`、`train6_4b.sub`、`make_dags_6.sh`、`jobs6_4b.dag`、`smoke6_4b.dag`、`joblist_6_4b.txt`、`README_6.md`。
`run_train5.sh` 和第 5 批的完全一样（同一个 md5），覆盖也没关系。如果你自己改过 `run_train5.sh`，就先把你的那份备份一下再解包。
不需要新容器，也不需要新模型：`/staging` 里有 `cotd_4b.sif` 和 `Qwen3.5-4B.tar` 就行。

## 1. 冒烟测试（先跑这个）
```bash
cd ~/cotd_chtc && condor_submit_dag smoke6_4b.dag
```
共 2 个小任务，每个大约 15–20 分钟：`smoke_154_4b_gsmsi_SIB`（逐步内化）和 `smoke_155_4b_gsmops_GFPQV`（数值节点）。
`results/results_smoke_*.tgz` 里的 `logs/*.out` 应该有：`preflight: gpu 0 … ok`、`code pin: a91ddbe…`、`smoke ok SIB` 或 `smoke ok GFPQV`、`=== smoke passed`。
请用打过补丁的包跑冒烟，然后把这 2 个 log 发给 Sora。

## 2. 提交
```bash
cd ~/cotd_chtc && condor_submit_dag jobs6_4b.dag
```
和第 5 批的 DAG 互不影响：节点名（j154…）、log 文件（`logs/train6_4b.log`）都不一样。失败的任务会自动重试一次，POST 脚本照旧把结果推到 GitHub。第 5 批正在跑的任务不用停。

## 3. 任务表和预计时间
估计方法和第 5 批一样：单卡 FSDP，0.25 h 固定开销 + 每次更新 2.2 s + 每百万训练 token 0.126 h（按 42、75、83–85 的实测时间拟合），针对 L40S 一类的卡，包含评测；H100/H200 大约快一倍。

| 编号 | 方法 | 数据 | 轮数 | 训练 tokens | 更新次数 | 预计 | 优先级 |
|---|---|---|---|---|---|---|---|
| 154 | SIB（逐步内化，从答案端删） | GSM8K si | 5 | 3.3M | 930 | 1.2 h | 130 |
| 155 | GFPQV（数值节点） | GSM8K ops | 4 | 约 8.6M | 741 | 1.8 h | 129 |
| 156 | GFPQO（选运算式） | GSM8K ops | 4 | 约 8.1M | 741 | 1.7 h | 128 |
| 157 | A2（只训答案） | GSM8K | 5 | 2.9M | 926 | 1.2 h | 127 |
| 158 | GFPQVQ（数值节点 + 子问题） | GSM8K ops | 4 | 约 8.5M | 741 | 1.8 h | 126 |
| 159 | SIYB（SIB + yes/no 节点） | GSM8K si | 5 | 15.2M | 930 | 2.7 h | 125 |
| 160 | SISB（SIB，错误数值，对照） | GSM8K si | 5 | 3.3M | 930 | 1.2 h | 124 |
| 161 | SI（逐步内化，从前面删） | GSM8K si | 5 | 3.3M | 930 | 1.2 h | 123 |
| 162 | GFPVB（数值题，不给题目） | GSM8K ops | 4 | 约 3.7M | 741 | 1.2 h | 122 |
| 163 | GFPQVM（yes/no + 数值节点） | GSM8K ops | 4 | 约 18M | 741 | 3.0 h | 121 |

合计约 17 卡时（单卡）。前四个（154–157）回答主要问题，一共约 6 卡时。全部任务都在 4 小时以内，`+GPUJobLength = "medium"` 足够。

## 出问题时
请把 `logs/<任务名>.condor.err` 和 `results/results_<任务名>.tgz` 里的 log 发给 Sora。如果是预检失败（log 里有 `GPU preflight FAILED on <机器>`），请顺便告诉我们是哪台机器。

## 第 5 批怎么处理（方向变了）
我们已经全面转向新方法，第 5 批大部分是旧方法的任务，不用再跑了：
- **保留**：101–104（GSM8K 是/否推理树和对照补种子），以及只训答案的轮数扫描 119、121、124、125、127、129、130、132、135、136、137、139、142、143。
- **不用跑**（还没跑的请 `condor_rm`）：105–118、120、122、123、126、128、131、133、134、138、140、141、144–153。
不好意思来回改，谢谢！
