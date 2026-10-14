# Gemma-4-12B 实验包（HTCondor）

这个包里有：训练代码（固定版本）、3 个任务的数据（GSM8K、数学链、K&K）、27 个训练任务的配置。
每个任务就是一次完整训练，跑完会自动评测。结果只有指标和预测文件，不保存模型权重。

## 0. 前提
- 解压到 `~/cotd`：`mkdir -p ~/cotd && tar xzf cotd_condor_bundle.tgz -C ~/cotd`
- 计算节点要能读写 `~/cotd`（共享文件系统）。如果不能，请告诉我，我改成 HTCondor 传文件的方式。
- 磁盘：大约 35 GB（Python 环境约 10 GB，模型 24 GB）。

## 1. 安装（一次，在能联网的登录节点上）
```bash
cd ~/cotd
bash condor/setup.sh        # google/gemma-4-12B 是 Apache-2.0、不需要登录或 token
```
需要 python3.11。没有的话，用 `module load` 或 conda 装一个 3.11。

## 1b. 结果自动推到 GitHub（一次）
每个任务跑完会自动把结果（指标和预测文件，几 MB，不含权重）推到一个私有仓库，Sora 那边直接就能看到。
```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/cotd_results -C "cotd-results-condor"
cat ~/.ssh/cotd_results.pub      # 把这一行公钥发给 Sora，由 Sora 加到仓库的 deploy key（只对这一个仓库有写权限）
```
Sora 加好以后，测试一下：`GIT_SSH_COMMAND="ssh -i ~/.ssh/cotd_results" git ls-remote git@github.com:tree-forcing-lab/cot-distillation-jev-results.git`
（22 端口不通的话会自动改走 443 端口。）没有这个 key 也能跑，结果留在 `~/cotd/runs/`，之后再推或用第 4 步打包。

## 2. 冒烟测试（先跑这个，几分钟）
```bash
cd ~/cotd && mkdir -p logs
condor_submit condor/smoke.sub
```
跑完后看 `logs/02_12b_gsm_GFPQ-c4lr3g12_s12_fsdp-smoke.out`，里面应该有 `smoke ok GFPQ` 和 `smoke ok base eval`。
把这个文件发给我确认一下。

## 3. 提交全部 27 个任务
```bash
cd ~/cotd && condor_submit condor/job.sub
```
- 每个任务申请 2 张 80 GB 以上的卡、12 个 CPU、230 GB 内存，预计 3–8 小时。
- `joblist.txt` 里的数字是优先级，越大越先跑。前 9 个最重要。
- 如果申请不到 230 GB 内存：把 `job.sub` 里的 `environment` 改成 `"REQUEST_CPUS=12 FSDP_OPTIM_OVERRIDE=8bit-sr"`，`request_memory` 改成 `100GB`（显存占用会稍大）。
- `requirements` 里的显存属性名（`GPUs_GlobalMemoryMb`）不同集群可能不一样，提交报错的话按你们集群的写法改。
- 跑完的任务会自动跳过：重新提交不会重复跑。

## 4. 发回结果
自动推送配好的话不用做任何事。没配好或推送失败时，二选一：
```bash
cd ~/cotd && bash condor/push_results.sh condor/jobs/<任务名>.env   # 手动补推某个任务
cd ~/cotd && bash condor/pack_results.sh                            # 或打包成 results_12b_<日期>.tgz 发给 Sora
```

## 出问题时
发我对应的 `logs/<任务名>.out` 和 `logs/condor_*.err`。
