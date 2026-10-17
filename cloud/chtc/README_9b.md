# Qwen3.5-9B 实验包（CHTC 第 4 批）

这一批是 7 个 Qwen3.5-9B 训练任务（GSM8K 4 个，算术链 25% 3 个），和第 3 批的 4B 任务配成同一族、同样后训练的扩展阶梯（4B → 9B）。
每个任务跑完自动评测，结果照旧推到 GitHub。和前几批共用 `~/cotd_chtc`（同一个 `chtc_settings.inc`、`post_push.sh`、`logs/`、`results/`），代码固定在 commit 60bf5a1（训练代码和第 3 批的 5a0460b 完全一样，只多了一个这批不用的脚本）。

## 0. 解包（一次）
```bash
tar xzf cotd_chtc_9b_bundle.tgz -C ~/cotd_chtc
cd ~/cotd_chtc && md5sum -c MANIFEST_9b.md5
```
- 全是新文件，不覆盖第 3 批和 12B 的任何东西。包里的 `run_train.sh` 和你现在的那份一字不差（md5 `e61eff1e…`），覆盖了也没有影响。
- 前提：第 3 批的 `cotd_4b.sif` 已经在 `/staging/<netid>/`。**这批不建新容器**，9B 和 4B 是同一族模型，`cotd_4b.sif` 里的 flash-linear-attention 直接能用。

## 1. 下载 9B 模型（一次，交互式 build job）
```bash
cd ~/cotd_chtc && condor_submit -i build_9b.sub
# 进入 build job 之后：
cp /staging/<netid>/cotd_4b.sif .
bash fetch_model_9b.sh /staging/<netid>    # 下载 Qwen/Qwen3.5-9B（固定版本，逐个文件校验）-> /staging/<netid>/Qwen3.5-9B.tar（约 19.3 GB）
exit
```
- 不需要 Hugging Face token。最后应打印 `Qwen3.5-9B ok at c202236…: … 4 safetensors shard(s), 19.3 GB of weights`。
- 下载完请把 `/staging/<netid>/Qwen3.5-9B.sha256` 发给 Sora。
- 开始前请看一下 `/staging` 的剩余空间够不够再放 19.3 GB（12B 和 4B 存的权重也在 `/staging/weights/`）。

## 2. 冒烟测试（先跑这个）
```bash
cd ~/cotd_chtc && condor_submit_dag smoke9b.dag
```
两个小任务（各 2 张卡，主要时间花在解压 19 GB 的模型上，约 20 分钟）：
- `results/results_smoke_62_9b_gsm_GFPQ.tgz`：解开里面的 `logs/*.out`，应该有 `smoke ok GFPQ` 和 `=== smoke passed`；
- `results/results_smoke_65_9b_math25_GFPS.tgz`：应该有 `smoke ok GFPS` 和 `=== smoke passed`。

把这两个 log 发给 Sora 确认。

## 3. 提交 7 个任务
```bash
cd ~/cotd_chtc && condor_submit_dag jobs9b.dag
```
- 可以和 `jobs.dag`、`jobs4b.dag` 同时跑（不同的 DAG、任务名、log 文件）。查看进度：`condor_q -dag`。失败的任务自动重试一次，POST 脚本推结果到 GitHub。
- 优先级 53–59，低于第 3 批的所有任务（60–75），排在已经提交的任务后面。顺序：GSM8K 三个（62、61、60）→ 63 → 算术链（65、64、66）。

## 资源
每个任务：2 张 **40 GB 以上**的卡（驱动 ≥ 12.8），12 CPU，**200 GB 内存**，60 GB 磁盘，`+GPUJobLength = "medium"`（24 小时以内）。
- 内存：FSDP + cpu-adamw 把 fp32 主权重和 AdamW 状态放在内存里，每个参数 12 字节，9B（约 9.65e9 个参数）约 116 GB；加载时两个进程各有一份 19 GB 的 bf16 模型。4B 实测每个进程 27 GiB，按参数量放大到 9B 约 112 GiB，峰值估计 160–175 GB，所以要 200 GB（12B 要 230 GB）。
- 显卡：4B 每张卡峰值 11–12 GiB，12B 约 28 GiB，9B 估计 20–25 GiB，40 GB 的卡够用。
- 磁盘：模型从 `/staging` 直接解压到任务目录（19.3 GB），不存权重（`SAVE` 关着）。

## 预计时间
假设：9B 每个 token 的计算量约为 4B 的 2.2 倍，GPU 速度和第 3 批的假设一样（两张 H100 约为我们原来集群的 3 倍）；训练 token 数和 4B 相同（同一个分词器）。
“预计”= 第 3 批 4B 的估计 × 2.2；“偏慢情况”另外考虑了 CPU 上的 AdamW：4B 在我们集群上实测每次更新 2.0–2.4 秒花在 CPU 优化器上，这部分不随显卡变快，9B 约为 2.07 倍，只训答案的任务受影响最大。分到 A100 80 GB 时 GPU 部分大约再乘 2。

| 任务 | 内容 | 训练 tokens | 更新次数 | 预计（H100） | 偏慢情况 |
|---|---|---|---|---|---|
| 60 | GSM8K 只训答案 12 轮 | 7.0M | 2,222 | 约 2.2 h | 约 3.5 h |
| 61 | GSM8K 只训答案 4 轮 | 2.3M | 741 | 约 0.8 h | 约 1.4 h |
| 62 | GSM8K Tree Forcing 4 轮 | 11.9M | 744 | 约 2.2 h | 约 2.3 h |
| 63 | GSM8K 对照（placebo）4 轮 | 约 12M | 744 | 约 2.2 h | 约 2.3 h |
| 64 | 算术链 25% 只训答案 12 轮 | 约 4.5M | 1,032 | 约 1.1 h | 约 1.7 h |
| 65 | 算术链 25% Tree Forcing 4 轮 | 约 18M | 344 | 约 1.5 h | 约 1.6 h |
| 66 | 算术链 25% 对照（placebo）4 轮 | 约 18M | 344 | 约 1.5 h | 约 1.6 h |

均含评测。合计约 12–15 个 2 卡小时。61 和算术链没有第 3 批的估计，按我们集群上 4B 的实测时间（61：0.8 h，64：1.1 h，65、66：1.7 h，只算训练）除以 3 再乘 2.2，加评测。

## 每个任务回答什么
- **扩展阶梯 4B → 9B**：同一族（Qwen3.5）、同样的后训练、同样的参数（学习率 3e-6、label smoothing 0.1、2 卡 FSDP、cpu-adamw、种子 12），只换模型。60、62 和第 3 批的 41、42 只差模型（以及 9B 不存权重）。
- **62 vs 60 / 61**：9B 上 Tree Forcing 是否仍然好过只训答案（12 轮：同样的训练时间；4 轮：同样的轮数）。
- **63 vs 62**：对照题换成关于题面的表面问题（placebo）。9B 上的提升来自节点内容，还是只是多看了几遍题。
- **64 / 65 / 66**：算术链（训练深度 ≤ 4，测试 d5–d8）上同样的三组。和 4B（我们集群上已跑完）、12B（任务 29、20、21，数据完全相同）放在一起，看深度外推随规模怎么变。
- 9B 还可以直接对照 StartLux 公布的 9B Instant 数字。
- 所有任务的 TAG 都带 `ch`（`c4lr3ch`、`c12lr3ch`），训练中都评测验证集（EVALMID 第一个是 val）。

## 出问题时
发 `logs/<任务名>.condor.err` 和 `results/results_<任务名>.tgz` 里的 log 给 Sora。
