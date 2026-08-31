# C-IDC 最佳参数组合 — ACC 均值 0.800

> 来源：贝叶斯搜索 Round 1（`sefs_bayes_opt_20260816_104936/`），第 1 轮参数组 01
> 结论：这是当前调参能达到的**最高均值**版本。方差为 72 样本小数据的固有特性（见 HANDOFF.md「四」节诊断）。

---

## 一、统计摘要

| 指标 | 均值 | 标准差 | 范围 |
|------|------|--------|------|
| ACC | **0.8000** | 0.0668 | 0.6806 ~ 0.9167 |
| ARI | 0.3589 | — | — |
| NMI | 0.2801 | — | — |

---

## 二、10 个 seed 的完整结果

| seed | ACC | ARI | NMI | Silhouette | DBI |
|:----:|:-----:|:-----:|:-----:|:----------:|:-----:|
| 0 | 0.7778 | 0.2756 | 0.2132 | 0.4159 | 1.2727 |
| 1 | 0.8056 | 0.3621 | 0.2638 | 0.2925 | 1.4369 |
| 2 | 0.8472 | 0.4708 | 0.3493 | 0.3896 | 1.1206 |
| 3 | 0.8472 | 0.4722 | 0.3554 | 0.5245 | 0.8542 |
| 4 | 0.7361 | 0.1779 | 0.1343 | 0.3698 | 1.4398 |
| 5 | 0.8194 | 0.3992 | 0.3136 | 0.3788 | 1.1346 |
| 6 | 0.7500 | 0.2399 | 0.1962 | 0.2906 | 1.4703 |
| 7 | 0.8194 | 0.3850 | 0.3006 | 0.2588 | 1.5380 |
| 8 | **0.9167** | 0.6881 | 0.5627 | 0.2573 | 1.4786 |
| 9 | 0.6806 | 0.1192 | 0.1115 | 0.3673 | 1.7507 |

> 最好 seed = 8（ACC 0.9167），最差 seed = 9（ACC 0.6806）。

---

## 三、参数组合

### 精确值（来自贝叶斯搜索）

| 参数 | 值 |
|------|-----|
| batch_size | 10 |
| correlation_threshold | 0.07658942254342552 |
| epochs | 1052 |
| ae_non_gated_epochs | 70 |
| ae_pretrain_epochs | 627 |
| start_global_gates_training_on_epoch | 750 |
| gates_hidden_dim | 151 |
| lr_pretrain | 0.0006355448050622945 |
| lr_clustering | 0.003751420984770231 |
| lr_aux_classifier | 9.918989909874554e-05 |
| sched_pretrain_min_lr | 2.0908316722254855e-05 |
| sched_clustering_min_lr | 3.1309446713093405e-07 |
| local_gates_lambda | 0.9249873521085907 |
| global_gates_lambda | 0.00012510871163802221 |
| gtcr_lambda | 0.04024097266067439 |
| tau | 68.99791892482523 |
| eps | 0.4866594919720787 |
| mask_percentage | 0.7190866650608354 |
| sefs_tau | 0.8705888981237058 |

### 可复现的 yaml 配置（写入 `cfg/ALLAML_best.yaml`）

```yaml
dataset: ALLAML
data_file: dataset/ALLAML.npz
correlation_threshold: 0.0766
batch_size: 10
seeds: 10
validate: true
epochs: 1052
ae_non_gated_epochs: 70
ae_pretrain_epochs: 627
start_global_gates_training_on_epoch: 750
mask_percentage: 0.719
latent_noise_std: 0.01
gtcr_loss: true
gtcr_projection_dim: null
gtcr_eps: 1
gtcr_lambda: 0.0402
eps: 0.487
use_gating: true
gates_hidden_dim: 151
encdec: [512, 512, 2048, 32, 2048, 512, 512]
clustering_head: [32, 128]
aux_classifier: [128]
tau: 69.0
sefs_tau: 0.871
local_gates_lambda: 0.925
global_gates_lambda: 0.000125
lr:
  pretrain: 6.355e-4
  clustering: 3.751e-3
  aux_classifier: 9.919e-5
sched:
  pretrain_min_lr: 2.091e-5
  clustering_min_lr: 3.131e-7
trainer:
  devices: 1
  accelerator: gpu
  max_epochs: 1052
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
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe train_evaluate.py --cfg cfg/ALLAML_best.yaml
```

**注意**：复现依赖当前代码版本（含 GTCR 行列式引理优化 + Cholesky 内存释放）。由于 GPU 浮点非确定性，复现结果在 0.80±0.01 范围内，无法保证逐位相同。
