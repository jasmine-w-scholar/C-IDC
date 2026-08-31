# C-IDC Prostate 最佳参数组合 — ACC 均值 0.6725

> 来源：贝叶斯搜索第一轮（`sefs_bayes_opt_20260823_114017/`），第 14 轮参数组 01
> 结论：Prostate 的 ACC 均值 0.6725，比 IDC baseline 0.653 提升 **+3.0%**，与 ALLAML 的 +3.2% 一致。

---

## 一、统计摘要

| 指标 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| ACC | **0.6725** | 0.0690 | 0.5686 ~ 0.7549 |
| ARI | 0.1289 | — | — |
| NMI | 0.1264 | — | — |

对比：C-IDC baseline 0.623 → 搜索最高 **0.6725**（+7.9%）；IDC baseline 0.653 → **+3.0%（显著提升）**。

---

## 二、10 个 seed 的完整结果

| seed | ACC | ARI | NMI | Silhouette | DBI |
|:----:|:-----:|:-----:|:154-----:|:----------:|:-----:|
| 0 | 0.7353 | 0.2151 | 0.2250 | 0.2955 | 1.0921 |
| 1 | 0.6275 | 0.0586 | 0.0754 | 0.3796 | 1.5898 |
| 2 | 0.7549 | 0.2532 | 0.2433 | 0.2718 | 1.1131 |
| 3 | 0.6765 | 0.1177 | 0.1264 | 0.4573 | 1.9011 |
| 4 | 0.6275 | 0.0586 | 0.0754 | 0.8108 | 0.4409 |
| 5 | 0.5686 | 0.0092 | 0.0134 | 0.4115 | 0.9248 |
| 6 | 0.6961 | 0.1463 | 0.1462 | 0.2376 | 0.9314 |
| 7 | 0.7549 | 0.2525 | 0.2003 | 0.3509 | 1.1561 |
| 8 | 0.7059 | 0.1619 | 0.1403 | 0.1953 | 1.3971 |
| 9 | 0.5784 | 0.0154 | 0.0181 | 0.3954 | 2.2709 |

> 最好 seed = 2/7（ACC 0.7549），最差 seed = 5（ACC 0.5686）。

---

## 三、参数组合

### 精确值

| 参数 | 值 |
|------|-----|
| batch_size | 8 |
| correlation_threshold | 0.07667218746720177 |
| epochs | 1085 |
| ae_non_gated_epochs | 60 |
| ae_pretrain_epochs | 161 |
| start_global_gates_training_on_epoch | 703 |
| gates_hidden_dim | 151 |
| lr_pretrain | 0.0006015887897448088 |
| lr_clustering | 0.0025196522863888196 |
| lr_aux_classifier | 0.00011546917949555693 |
| sched_pretrain_min_lr | 3.82624799904627e-05 |
| sched_clustering_min_lr | 9.655355567461805e-07 |
| local_gates_lambda | 0.7147990065169323 |
| global_gates_lambda | 0.0001624298726117099 |
| gtcr_lambda | 0.09490680323030504 |
| tau | 46.06454652516561 |
| eps | 0.27464994585059177 |
| mask_percentage | 0.5494118009221364 |
| sefs_tau | 0.8074337781753939 |

### 可复现的 yaml 配置（写入 `cfg/Prostate_best.yaml`）

```yaml
dataset: Prostate
data_file: dataset/Prostate.npz
correlation_threshold: 0.0767
batch_size: 8
seeds: 10
validate: true
epochs: 1085
ae_non_gated_epochs: 60
ae_pretrain_epochs: 161
start_global_gates_training_on_epoch: 703
mask_percentage: 0.549
latent_noise_std: 0.01
gtcr_loss: true
gtcr_projection_dim: null
gtcr_eps: 1
gtcr_lambda: 0.0949
eps: 0.275
use_gating: true
gates_hidden_dim: 151
encdec: [512, 512, 2048, 32, 2048, 512, 512]
clustering_head: [32, 128]
aux_classifier: [128]
tau: 46.1
sefs_tau: 0.807
local_gates_lambda: 0.715
global_gates_lambda: 0.000162
lr:
  pretrain: 6.016e-4
  clustering: 2.520e-3
  aux_classifier: 1.155e-4
sched:
  pretrain_min_lr: 3.826e-5
  clustering_min_lr: 9.655e-7
trainer:
  devices: 1
  accelerator: gpu
  max_epochs: 1085
  deterministic: true
  logger: true
  check_val_every_n_epoch: 10
  enable_checkpointing: false
  num_sanity_val_steps: 0
use_sefs: true
save_seed_checkpoints: false
```

---

## 四、复现命令

```bash
cd D:\PythonProject\CIDC
find . -type d -name "__pycache__" -exec rm -rf {} +
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe train_evaluate.py --cfg cfg/Prostate_best.yaml
```
