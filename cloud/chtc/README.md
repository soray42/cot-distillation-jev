# Gemma-4-12B 实验包（CHTC 版）

和上一版的区别：CHTC 的计算节点读不到登录服务器的文件，所以这一版改成：
- 环境装在 Apptainer 容器里；
- 模型和容器都放在 /staging；
- 代码和数据每个任务传过去（20 MB）；
- 结果传回登录服务器，再自动推到 GitHub。

全部在 `/home/<netid>/cotd_chtc` 下操作：`mkdir -p ~/cotd_chtc && tar xzf cotd_chtc_bundle.tgz -C ~/cotd_chtc`

## 0. 设置（一次）
编辑 `chtc_settings.inc`，把 `/staging/CHANGE_ME` 改成你的 /staging 目录。

## 1. 建容器、下载模型（一次，用交互式 build job）
```bash
cd ~/cotd_chtc && mkdir -p logs results
condor_submit -i build.sub
# 进入 build job 之后：
apptainer build cotd.sif cotd.def          # Python 3.11 + torch 2.11 (CUDA 12.8) + transformers 5.17，约 15 分钟
mv cotd.sif /staging/<netid>/
cp /staging/<netid>/cotd.sif .             # 下一步要用
bash fetch_model.sh /staging/<netid>       # 下载 google/gemma-4-12B（固定版本，sha256 校验）-> /staging/<netid>/gemma-4-12B.tar
exit
```
- 选 torch 2.11 + cu128，是因为大部分 80 GB 机器的驱动是 CUDA 12.8；4B 用的 torch 2.14 只有 CUDA 13 版本。
- 这个模型不需要 Hugging Face token。

## 2. 结果自动推到 GitHub（一次，在登录服务器上）
```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/cotd_results -C "cotd-results-chtc"
cat ~/.ssh/cotd_results.pub      # 把这一行公钥发给 Sora，由 Sora 加到结果仓库的 deploy key（只对这一个仓库有写权限）
```
Sora 加好以后测试一下：`GIT_SSH_COMMAND="ssh -i ~/.ssh/cotd_results" git ls-remote git@github.com:tree-forcing-lab/cot-distillation-jev-results.git`
22 端口不通的话会自动改走 443 端口。没有这个 key 也能跑，结果都在 `results/` 里。

## 3. 冒烟测试（先跑这个）
```bash
cd ~/cotd_chtc && condor_submit_dag smoke.dag
```
跑完后看 `results/results_smoke_02_12b_gsm_GFPQ.tgz`，解开里面的 `logs/*.out`，应该有 `smoke ok GFPQ` 和 `smoke ok base eval`。
把这个 log 发给 Sora 确认。

## 4. 提交全部 27 个任务
```bash
cd ~/cotd_chtc && condor_submit_dag jobs.dag
```
- 每个任务：2 张 80 GB 以上、驱动 ≥ 12.8 的卡，12 CPU，230 GB 内存，80 GB 磁盘，`+GPUJobLength = "medium"`（24 小时以内，预计 3–8 小时）。
- DAG 里的 PRIORITY 越大越先跑，前 9 个最重要。失败的任务会自动重试一次。
- 每个任务结束后，DAG 的 POST 脚本会把结果推到 GitHub（`post_push.sh`）。
- 查看进度：`condor_q -dag`；DAG 中断后再次运行 `condor_submit_dag jobs.dag` 会自动续跑（rescue DAG）。

## 出问题时
发 `logs/<任务名>.condor.err` 和 `results/results_<任务名>.tgz` 里的 log 给 Sora。
