# Qwen3.5-4B 实验包（CHTC 第 3 批）

这一批是 16 个 Qwen3.5-4B 训练任务（14 个现在提交，2 个先 hold）（GSM8K 和 MuSiQue），每个任务跑完自动评测，结果照旧推到 GitHub。
和 12B 共用 `~/cotd_chtc`（同一个 `chtc_settings.inc`、`post_push.sh`、`logs/`、`results/`），代码固定在 commit 5a0460b。

## 0. 解包（一次）
```bash
tar xzf cotd_chtc_4b_bundle.tgz -C ~/cotd_chtc
cd ~/cotd_chtc && md5sum -c MANIFEST_4b.md5
```
- 只替换一个旧文件：`run_train.sh`。新版本对 12B 任务的行为完全不变（还是用 `cotd_code_data.tgz` 和 `gemma-4-12B.tar`），
  正在跑或排队的 12B 任务不用动。其余都是新文件，不会覆盖 `chtc_settings.inc` 和 `post_push.sh`。

## 1. 建 4B 容器、下载 4B 模型（一次，交互式 build job）
```bash
cd ~/cotd_chtc && condor_submit -i build_4b.sub
# 进入 build job 之后：
cp /staging/<netid>/cotd.sif .
apptainer build cotd_4b.sif cotd_4b.def    # 在 cotd.sif 上加装 flash-linear-attention 0.5.2，几分钟；最后应打印两行 "Qwen3.5 ... -> fla...."
bash fetch_model_4b.sh /staging/<netid>    # 下载 Qwen/Qwen3.5-4B（固定版本，逐个文件校验）-> /staging/<netid>/Qwen3.5-4B.tar（约 9 GB）
mv cotd_4b.sif /staging/<netid>/
exit
```
- 为什么要新容器：Qwen3.5 的线性注意力层要用 flash-linear-attention，没有它会退回慢 10 倍以上的 PyTorch 实现；12B 的 `cotd.sif` 里没有装它。
  12B 任务继续用 `cotd.sif`，不受影响。
- 这个模型不需要 Hugging Face token。下载完请把 `/staging/<netid>/Qwen3.5-4B.sha256` 发给 Sora。

## 2. 冒烟测试（先跑这个）
```bash
cd ~/cotd_chtc && condor_submit_dag smoke4b.dag
```
三个小任务（主要时间花在解压模型上，十几分钟）：
- `results/results_smoke_46_4b_musique25_GFPQ.tgz`（2 张卡）：解开里面的 `logs/*.out`，应该有 `smoke ok GFPQ` 和 `=== smoke passed`；
- `results/results_smoke_47_4b_musique25_GFPQX.tgz`（2 张卡）：应该有 `smoke ok GFPQX` 和 `=== smoke passed`；
- `results/results_smoke_44_4b_gsmrat_A7.tgz`（1 张卡）：应该有 `smoke ok A7` 和 `=== smoke passed`。

把这三个 log 发给 Sora 确认。

## 3. 提交 14 个任务（46、47 先不交）
```bash
cd ~/cotd_chtc && condor_submit_dag jobs4b.dag
```
- 可以和 12B 的 `jobs.dag` 同时跑（不同的 DAG、不同的任务名、不同的 log 文件）。查看进度：`condor_q -dag`。
- 资源：41–43、45–49 每个任务 2 张 **40 GB 以上**的卡（4B 每张卡峰值只有约 12 GB）、12 CPU、120 GB 内存；44 和 50 各要 1 张 80 GB 以上的卡、8 CPU、80 GB 内存。磁盘 40 GB，`+GPUJobLength = "medium"`（24 小时以内）。
- 优先级：41、42（对照，各约 1 小时）和 46、47（MuSiQue 推理树和对照）排在最前面（71–75），其余 56–61。MuSiQue 是我们现在最缺的结果，所以插在 12B 第二批中间。
- 41、42 是对照任务（和我们原来集群上的结果比），排在最前面。50 是 44（A7）的同配置对照：同样 1 张卡、同样精度，只是不训推理文本。失败的任务自动重试一次，POST 脚本推结果到 GitHub，和 12B 一样。

## 2026-10-17 第二次更新（按 Milo 的建议）
- **新加 4 个 GSM8K 任务**（每个约 1 小时）：55 打乱节点答案（GFPQR）、56 只保留前 3 步的节点（数据 `student_gsm8k_tf_s3`，丢掉 2,466 / 23,692 个节点，val 不变）、57 只训答案学习率 1e-5、58 推理树 25% 数据 12 轮。
- **41、42 保存权重**（`SAVE=1`，每个约 8.4 GB，进 `$STAGING/weights/`），给探针用。
- **所有任务（含推理树）训练中都评测验证集**，挑轮数时两边对等。
- **46、47（MuSiQue 推理树和对照）先 hold**：MuSiQue 的错误节点有一半类型不对（你查出来的），正在按答案类型重新生成。修好后我会单独发一个新的代码包和 `jobs4b_hold.dag`。48（MuSiQue 只训答案 4 轮）排在最前面，先看只训答案是不是已经饱和。45、48 只用最终题，不受节点修改影响。
- `jobs4b.dag` 里现在是 14 个任务；`jobs4b_hold.dag` 是被 hold 的两个，**现在不要提交**。

## 预计时间
假设：两张 H100 上约 6,000 tokens/s，是我们原来集群（两块 A100 的一半，4g.40gb）约 2,000 tokens/s 的 3 倍；如果分到的是 A100 80 GB，大约再乘 2。
下表含评测，按训练 token 数和我们集群上的实测时间除以 3 估算：

| 任务 | 内容 | 训练 tokens | 我们集群 | 预计（H100） |
|---|---|---|---|---|
| 41 | GSM8K 只训答案 12 轮 | 7.0M | 2.4 h | 约 1 h |
| 42 | GSM8K Tree Forcing 4 轮 | 11.9M | 1.9 h | 约 1 h |
| 43 | GSM8K 只训答案 20 轮 | 11.7M | 约 4 h | 约 1.5 h |
| 44 | GSM8K 推理文本辅助（A7，1 张卡）12 轮 | 20.6M | 没跑过 | 2–4 h |
| 45 | MuSiQue 只训答案 12 轮 | 80M | 约 11 h | 约 4 h |
| 46 | MuSiQue Tree Forcing 4 轮 | 92M | 约 13 h | 约 4.5 h |
| 47 | MuSiQue 对照（placebo）4 轮 | 约 92M | 约 13 h | 约 4.5 h |
| 48 | MuSiQue 只训答案 4 轮 | 27M | 约 4 h | 约 1.5 h |
| 49 | GSM8K 25% 对照（placebo）4 轮 | 3M | 0.5 h | 约 0.3 h |
| 50 | GSM8K 只训答案 12 轮（1 张卡，和 44 同配置） | 7.0M | 约 2.5 h | 约 1–1.5 h |
| 53、54 | GSM8K 25% 只训答案 4 轮（种子 12、13），训练中评测验证集，用来按验证集挑轮数 | 0.6M | 约 0.3 h | 约 0.2 h |

最重的是 45、46、47（MuSiQue，每条约 1,400 token）。合计约 23 个 2 卡小时。只训答案的任务在训练中都会评测验证集（EVALMID 里加了 val）。所有 4B 任务的 TAG 都带 `ch`（单卡的是 `ch1`），和我们原来集群上的结果分开。

## 出问题时
发 `logs/<任务名>.condor.err` 和 `results/results_<任务名>.tgz` 里的 log 给 Sora。
