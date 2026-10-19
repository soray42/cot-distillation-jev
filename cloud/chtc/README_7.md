# CHTC 第 7 批：所有方法改跑 1 轮和 2 轮，Qwen3.5-4B，18 个任务

你那边的结果说明，只训答案（A2）在 GSM8K 4B 上 1 轮最好（种子 12/13/14 的测试分数是 70.5 / 69.7 / 68.8，2 轮 69.8，4 轮约 67.8）。之前的方法都是 4–5 轮跑的，已经过了最好的点。所以这批把方法都改成 1 轮和 2 轮，另外补你的多选节点（MC）的种子，再加一个新方法：按解题步骤的课程学习（GFPQC）。

**请把这批排在最前面。** DAG 里的优先级是 160–143（第 6 批是 130–121），所以和别的 DAG 一起排队时这批先跑。

| 包 | 模型 | 任务 | 代码 |
|---|---|---|---|
| `cotd_chtc_7_4b_bundle.tgz` | Qwen3.5-4B | 18 个（201–218） | 包里 `CODE_PIN` 文件写的 commit |

编号从 201 开始，跳过了 185–200，免得和你自己的任务（164–184 及之后）撞号。run 目录全是新的，`runs/` 里已有的结果不会被覆盖。设置和第 5、6 批的 4B 任务一样：全参数微调，FSDP bf16 + cpu-adamw，学习率 3e-6，不存权重。

## 最重要的一点：201、202、203、205 需要你的 MC 代码
这 4 个是你的多选节点 GFPQ（你那边叫 GFPQM，182 / 183 那种，run 目录 `GFPQ-c1lr3mc-…`、`GFPQ-c2lr3mc-…`），只是种子换成 13 和 14。参数是从你 182 / 183 的 `metrics.json` 里抄的：GFPQ 加 `--node-format mc`，别的完全一样。

但这个 MC 格式是你自己补丁里的（code pin `5a0460b…-fsdp1-v5`），我们的代码里没有：我们的 `train_student.py` 遇到 GFPQ 的视图加 `--node-format mc` 会直接报错退出。所以请二选一：
- **(a)** 打 fsdp1 补丁的时候，把你的 MC 补丁也一起打进 `cotd_code_data_7_4b.tgz`。这样 18 个任务都用这一个包，冒烟任务 `smoke_201` 会检查补丁是否已经打进去。
- **(b)** 如果 MC 补丁打不进新代码，就用你自己的 v5-ep 包跑这 4 个（和 182 / 183 一样，只改 `SEED` / `ORDERSEED`，具体设置见 `condor/jobs7_4b/201…205` 的 env 文件），优先级还是 160、159、158、156。然后 `jobs7_4b.dag` 里就不要提交 j201、j202、j203、j205；冒烟也只跑 `smoke_207`（`smoke_201` 用我们的包一定会失败，可以忽略）。

另外，方便的话请把 MC 补丁的 diff 发给 Sora，我们想把它合进仓库。

## 这批是什么
- **MC 节点种子**（201–203、205）和**只训答案 2 轮的种子 13 / 14**（204、206）：目前唯一的正结果就是 MC 节点 1 / 2 轮，这 6 个用来确认它，所以排在最前面。
- **新方法 1 轮和 2 轮**，种子 12：
  - SIYB、SIB（逐步内化，数据 `student_gsm8k_tf_si`）：还是 4 + 1 个阶段，但 `EPOCHS` 现在是**总**轮数，1 轮就是每个阶段 0.2 轮。
  - GFPQVM、GFPQV、GFPQO（数值节点，数据 `student_gsm8k_tf_ops`）。
- **GFPQC（新，按解题步骤的课程学习**，数据 `student_gsm8k_tf`）：先只训第 1 步的 yes/no 节点题，再训第 1–2 步、第 1–3 步，然后是所有步骤，最后只训原来的最终问题。一共 5 个阶段，每个阶段占总轮数的 1/5。AdamW 状态在阶段之间保留，学习率是一条贯穿所有阶段的 warmup + cosine。
- SI 和 GFPQC 的任务会在每个阶段结束时自己评测，所以 env 里的 `EVALAT` 是空的，这是故意的，不是漏了。

## 关于 4B 单卡 FSDP（和第 5、6 批一样）
- 请像之前那样，把你的 fsdp1 补丁打进 `cotd_code_data_7_4b.tgz`（需要的话 MC 补丁也一起打，见上面），每个任务用 1 张卡。
- `joblist_7_4b.txt` 第三列**已经是 1**，`jobs7_4b.dag` / `smoke7_4b.dag` 也已经按 1 卡生成好了（8 CPU、80 GB 内存、卡 ≥ 40 GB），这次**不用**再运行 `make_dags_7.sh`。env 里仍然是 `NGPU=2`，由你的补丁处理。
- 代码从第 6 批的 a91ddbe 换成了新的 pin。`hpc/train_arm.sbatch` 里 NGPU / torchrun 那一段没有改，只在 `arm_config` 里加了一行 GFPQC。`scripts/train_student.py` 加了 `--curriculum-steps`：参数、检查、`fulltree_updates` 多一个参数、阶段计划、学习率调度的条件和训练循环的条件。`src/cotdistill/student.py` 只是在 `si_group` 后面加了两个函数。如果你的补丁改到这几处附近，可能需要手动合并一下。
- 如果不打补丁：把 joblist 第三列改成 2，再运行 `bash make_dags_7.sh`（2 卡：12 CPU、120 GB）。

## 0. 解包
```bash
cd ~/cotd_chtc && tar xzf cotd_chtc_7_4b_bundle.tgz && md5sum -c MANIFEST_7_4b.md5
```
新文件有：`cotd_code_data_7_4b.tgz`、`train7_4b.sub`、`make_dags_7.sh`、`jobs7_4b.dag`、`smoke7_4b.dag`、`joblist_7_4b.txt`、`README_7.md`。`run_train5.sh` 和第 5、6 批的完全一样（md5 相同）。不需要新容器，也不需要新模型。

## 1. 冒烟测试（先跑这个，用打过补丁的包）
```bash
cd ~/cotd_chtc && condor_submit_dag smoke7_4b.dag
```
一共 2 个小任务，每个大约 15–20 分钟：
- `smoke_201_4b_gsm_GFPQM`（MC 节点）：检查 MC 补丁是否在包里。没打 MC 补丁的话，会马上报 `SMOKE FAILED GFPQ`，日志里有 `does not combine with … --node-format mc`。
- `smoke_207_4b_gsmsi_SIYB`（逐步内化 + yes/no 节点）。

`results/results_smoke_*.tgz` 里的 `logs/*.out` 应该有：`preflight: gpu 0 … ok`、`code pin: …`、`smoke ok GFPQ` 或 `smoke ok SIYB`、`=== smoke passed`。请把这 2 个 log 发给 Sora。

## 2. 提交
```bash
cd ~/cotd_chtc && condor_submit_dag jobs7_4b.dag
```
和前几批的 DAG 互不影响：节点名（j201…）、log 文件（`logs/train7_4b.log`）都不一样。POST 脚本照旧把结果推到 GitHub。

## 3. 任务表和预计时间
预计时间是按第 6 批和你 125–128、182–183 的实测时间，在 L40 / L40S 一类的卡上按轮数折算的，包含评测。

| 编号 | 方法 | 数据 | 轮数 | 预计 | 优先级 |
|---|---|---|---|---|---|
| 201 | GFPQ MC 节点，种子 13（你的代码） | GSM8K | 1 | 0.5 h | 160 |
| 202 | GFPQ MC 节点，种子 14（你的代码） | GSM8K | 1 | 0.5 h | 159 |
| 203 | GFPQ MC 节点，种子 13（你的代码） | GSM8K | 2 | 1.1 h | 158 |
| 204 | A2 只训答案，种子 13 | GSM8K | 2 | 0.6 h | 157 |
| 205 | GFPQ MC 节点，种子 14（你的代码） | GSM8K | 2 | 1.1 h | 156 |
| 206 | A2 只训答案，种子 14 | GSM8K | 2 | 0.6 h | 155 |
| 207 | SIYB（逐步内化 + yes/no 节点） | GSM8K si | 1（5 阶段） | 0.8 h | 154 |
| 208 | SIYB | GSM8K si | 2（5 阶段） | 1.3 h | 153 |
| 209 | SIB（逐步内化，从答案端删） | GSM8K si | 1（5 阶段） | 0.4 h | 152 |
| 210 | SIB | GSM8K si | 2（5 阶段） | 0.7 h | 151 |
| 211 | GFPQVM（yes/no + 数值节点） | GSM8K ops | 1 | 0.9 h | 150 |
| 212 | GFPQVM | GSM8K ops | 2 | 1.6 h | 149 |
| 213 | GFPQV（数值节点） | GSM8K ops | 1 | 0.5 h | 148 |
| 214 | GFPQV | GSM8K ops | 2 | 0.8 h | 147 |
| 215 | GFPQO（选运算式） | GSM8K ops | 1 | 0.5 h | 146 |
| 216 | GFPQO | GSM8K ops | 2 | 0.9 h | 145 |
| 217 | GFPQC（按步骤的课程学习） | GSM8K | 5（5 阶段，每阶段 1 轮；最后一阶段 = 只训答案 1 轮） | 约 2 h | 144 |
| 218 | GFPQC | GSM8K | 10（5 阶段，每阶段 2 轮；最后一阶段 = 只训答案 2 轮） | 约 4 h | 143 |

合计约 14 卡时（单卡）。前 6 个（201–206）大约 4.4 卡时。全部都在 4 小时以内，`+GPUJobLength = "medium"` 足够。

## 出问题时
请把 `logs/<任务名>.condor.err` 和 `results/results_<任务名>.tgz` 里的 log 发给 Sora。如果是预检失败（log 里有 `GPU preflight FAILED on <机器>`），请顺便告诉我们是哪台机器。谢谢！
