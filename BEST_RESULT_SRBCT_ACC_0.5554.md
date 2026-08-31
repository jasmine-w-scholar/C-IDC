# C-IDC SRBCT 最佳参数组合 — ACC 均值 0.5554

> 来源：贝叶斯搜索第一轮（`sefs_bayes_opt_20260820_221341/`），第 10 轮参数组 02
> 结论：SRBCT 的 ACC 均值天花板 ≈ 0.55，0.5554 已接近上限。对比 IDC baseline 0.554 仅 +0.25%，几乎持平。

---

## 一、统计摘要

| 指标 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| ACC | **0.5554** | 0.0728 | 0.4699 ~ 0.7229 |
| ARI | 0.1909 | — | — |
| NMI | 0.2798 | — | — |

对比：C-IDC baseline 0.506 → 搜索最高 **0.5554**（+9.7%）；IDC baseline 0.554 → **+0.25%（几乎持平）**。

---

## 二、10 个 seed 的完整结果

| seed | ACC | ARI | NMI | Silhouette | DBI |
|:----:|:-----:|:-----:|:-----:|:----------:|:-----:|
| 0 | 0.5904 | 0.1937 | 0.3022 | 0.2751 | 1.4919 |
| 1 | 0.5422 | 0.2048 | 0.2720 | 0.1837 | 1.6599 |
| 2 | 0.4699 | 0.1197 | 0.2079 | 0.1480 | 1.9202 |
| 3 | 0.4819 | 0.0906 | 0.1702 | 0.1395 | 2.0487 |
| 4 | **0.7229** | 0.3744 | 0.4247 | 0.1899 | 1.6501 |
| 5 | 0.5783 | 0.2410 | 0.3717 | 0.1957 | 1.6212 |
| 6 | 0.4940 | 0.0951 | 0.1683 | 0.1566 | 2.1608 |
| 7 | 0.5663 | 0.2611 | 0.3559 | 0.1414 | 2.0475 |
| 8 | 0.5301 | 0.1434 | 0.2375 | 0.2266 | 1.5688 |
| 9 | 0.5783 | 0.1857 | 0.2871 | 0.3026 | 1.4141 |

> 最好 seed = 4（ACC 0.7229），最差 seed = 2（ACC 0.4699）。

---

## 三、参数组合

### 精确值

| 参数 | 值 |
|------|-----|
| batch_size | 5 |
| correlation_threshold | 0.08815400748997464 |
| epochs | 587 |
| ae_non_gated_epochs | 43 |
| ae_pretrain_epochs | 221 |
| start_global_gates_training_on_epoch | 556 |
| gates_hidden_dim | 118 |
| lr_pretrain | 0.00042448548465955554 |
| lr_clustering | 0.003996947297759794 |
| lr_aux_classifier | 9.419805956935233e-05 |
| sched_pretrain_min_lr | 2.5144417288047166e-05 |
| sched_clustering_min_lr | 3.229233998724256e-07 |
| local_gates_lambda | 0.543290475209061 |
| global_gates_lambda | 0.000199142817857107 |
| gtcr_lambda | 0.07838762072722773 |
| tau | 50.59809146071082 |
| eps | 0.12564834844248224 |
| mask_percentage | 0.5670044376306114 |
| sefs_tau | 1.3384184573092033 |

### 可复现的 yaml 配置（写入 `cfg/SRBCT_best.yaml`）

```yaml
dataset: SRBCT
data_file: dataset/SRBCT.npz
correlation_threshold: 0.088
batch_size: 5
seeds: 10
validate: true
epochs: 587
ae_non_gated_epochs: 43
ae_pretrain_epochs: 221
start_global_gates_training_on_epoch: 556
mask_percentage: 0.567
latent_noise_std: 0.01
gtcr_loss: true
gtcr_projection_dim: null
gtcr_eps: 1
gtcr_lambda: 0.078
eps: 0.126
use_gating: true
gates_hidden_dim: 118
encdec: [512, 512, 2048, 32, 2048, 512, 512]
clustering_head: [32, 128]
aux_classifier: [128]
tau: 50.6
sefs_tau: 1.338
local_gates_lambda: 0.543
global_gates_lambda: 0.000199
lr:
  pretrain: 4.245e-4
  clustering: 3.997e-3
  aux_classifier: 9.420e-5
sched:
  pretrain_min_lr: 2.514e-5
  clustering_min_lr: 3.229e-7
trainer:
  devices: 1
  accelerator: gpu
  max_epochs: 587
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
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe train_evaluate.py --cfg cfg/SRBCT_best.yaml
```
