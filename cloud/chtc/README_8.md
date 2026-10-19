# CHTC 第 8 批：Tree Forcing v2（隐式节点槽位），Qwen3.5-4B，13 个任务

**这是什么（两句话）：** 每道题的题干和选项之间插入 K 个槽位 token（词表里已有的 `<|box_start|>`），训练时要求每个槽位的末层向量，作为一个单独短序列的开头，再经过同一个模型和同一个输出层，能还原出这道题的第 j 步算式（如 "4 + 2 = 6"）；损失 = 选项 CE + λ × 槽位解码 CE。部署时模型不变，一次前向，读法和 A2 一样（槽位只是 prompt 里多出来的几个 token）。

**请把这批排在最前面：** 优先级 190–178（第 7 批是 160–143）。编号 219–231，run 目录全是新的（`runs/LS*`），不会覆盖已有结果。设置和第 5–7 批的 4B 一样：全参数微调，FSDP bf16 + cpu-adamw，学习率 3e-6，种子 12，GSM8K 全量（`data/student_gsm8k_tf_si`），不存权重。

## 1. 先跑冒烟，把 log 发给 Sora 之后再提交其余任务
```bash
cd ~/cotd_chtc && tar xzf cotd_chtc_8_4b_bundle.tgz && md5sum -c MANIFEST_8_4b.md5
condor_submit_dag smoke8_4b.dag
```
2 个小任务（`smoke_219_4b_gsmsi_LS`、`smoke_225_4b_gsmsi_LSY`），各约 15–20 分钟。**这是槽位解码（Qwen3.5 走 `inputs_embeds` 的那条路径）第一次在 GPU 上跑**，所以请先把 `results/results_smoke_*.tgz` 里的 `logs/*.out` 发给 Sora，确认没问题后再提交：
```bash
condor_submit_dag jobs8_4b.dag
```
log 里应该有：`preflight: gpu 0 … ok`、`code pin: …`、`smoke ok LS` / `smoke ok LSY`、`=== smoke passed`。注意：冒烟只测训练和槽位解码，必要性测试（swap / drop / noise）要到正式任务结束时才会跑。

## 2. 单卡 FSDP 补丁（新代码上要重新打）
- 请像之前一样把 fsdp1 补丁打进 `cotd_code_data_8_4b.tgz`，每个任务 1 张卡。joblist 第三列已经是 1，DAG 已按 1 卡生成好（8 CPU、80 GB、卡 ≥ 40 GB），**不用**再运行 `make_dags_8.sh`。env 里仍是 `NGPU=2`，由你的补丁处理。
- 代码从第 7 批的 401ca5a 换成 cb0e84f。和你的补丁可能有冲突的地方：
  - `src/cotdistill/fsdp.py`：`_Call` 多了 `fn` 字段，`Readout.forward` 按 `call.fn` 分派，新增 `Readout.run(fn, ...)`（槽位解码那一遍就从这里在 FSDP 根模块的 forward 里跑）。
  - `hpc/train_arm.sbatch`：NGPU / torchrun 那一段**没改**；`arm_config` 里加了 `SL` 和 LS / LSU / LSR / LSY 四行；训练循环里 `OUT=` 那一行多了槽位后缀 `${SLD}`。
  - `scripts/train_student.py`：新增 `--latent-slots` 等参数和检查，FSDP 训练循环里有槽位的那一遍（`slot_forward`），训练后跑 `slot_evals`。
- **新的可执行文件 `run_train8.sh`**（`train8_4b.sub` 用它）：和 `run_train5.sh` 只差一处——打包结果时用 sbatch 实际写的 run 目录。变体的目录带后缀（`-sc2`、`-sc4`、`-spafter`、`-sl0.3`、`-sl3`），用 `run_train5.sh` 的话这 5 个任务的结果不会进 tgz。

## 3. 任务表（预计时间：单卡 L40/L40S，含评测；槽位解码那一遍在 GPU 上还没测过速，可能有 ±30% 误差）

| 编号 | 方法 | 轮数 | 变体 | 优先级 | 预计 |
|---|---|---|---|---|---|
| 219 | LS（槽位 + 解码监督） | 1 | 默认 K=6, c=1, λ=1, 选项前 | 190 | 0.7 h |
| 220 | LS | 2 | 默认 | 189 | 1.0 h |
| 221 | LSU（槽位不加监督，暂停 token 对照） | 1 | 默认 | 188 | 0.5 h |
| 222 | LSU | 2 | 默认 | 187 | 0.7 h |
| 223 | LSR（槽位目标来自别的题，错误内容对照） | 1 | 默认 | 186 | 0.7 h |
| 224 | LSR | 2 | 默认 | 185 | 1.0 h |
| 225 | LSY（LS + GFPQ 的是/否节点题） | 1 | 默认 | 184 | 0.9 h |
| 226 | LSY | 2 | 默认 | 183 | 1.5 h |
| 227 | LS | 2 | c=2（`SLOTC=2`） | 182 | 1.0 h |
| 228 | LS | 2 | c=4（`SLOTC=4`） | 181 | 1.1 h |
| 229 | LS | 2 | 槽位放在选项后（`SLOTPLACE=after-options`） | 180 | 1.0 h |
| 230 | LS | 1 | λ=0.3（`SLOTLAM=0.3`） | 179 | 0.7 h |
| 231 | LS | 1 | λ=3（`SLOTLAM=3`） | 178 | 0.7 h |

合计约 11.5 卡时，每个都在 2 小时以内。

## 4. 要看的输出（`runs/<run 目录>/`）
- `metrics.json` 里的 `slot_necessity`：在 gsm8kv2 和 gsm1k 上，`swap`（槽位 token 换成 `<|quad_start|>`）、`drop`（去掉槽位行）、`noise@mid:1.0`（中间层给槽位加噪声）三种扰动下的 `acc`、`acc_intact`、`acc_drop`（= 完整 − 扰动）。掉得多说明模型真的在用槽位。
- `slot_decode_val.jsonl`：val 前 32 题每个槽位解码出来的文本和目标算式；`metrics.json` 里 `slot_decode` 是汇总（exact / value 匹配率）。还有 `latent_slots`（槽位设置）和 `preds_<set>@<mode>.jsonl`。

出问题时请把 `logs/<任务名>.condor.err` 和 `results/results_<任务名>.tgz` 里的 log 发给 Sora。谢谢！
