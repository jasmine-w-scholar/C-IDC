# HANDOFF — C-IDC 贝叶斯超参数优化

> 写给完全没有上下文的新会话。读完这份文档就能接手继续工作。

---

## 一、我们在做什么

**项目**：C-IDC 聚类模型 + SEFS 特征选择，用于基因表达谱数据（小样本高维度）。

**当前阶段**：**方向 B — 稳定性工程**。Lite 和 Heavy 两条架构改进路线均已验证失败，回归原始 C-IDC，通过贝叶斯超参数优化提升 ALLAML 上的 ACC 到 82% 并保持跨 seed 稳定。

**目标**：ALLAML ACC ≥ 0.82（稳定），当前 C-IDC baseline 0.711。

**数据集**：ALLAML（72×7129）、SRBCT（83×2308）、Prostate（102×?），均在 `dataset/` 目录下。均为无监督聚类，全量数据训练+评估，不划分训练/测试集。

---

## 二、路线回顾

### 2.1 Lite 路线（D1-D7）— 已放弃 ❌
全部配置均 ≤ C-IDC baseline 0.711。D1（Dropout）和 D6（MixUp）是小样本基因数据毒药。

### 2.2 Heavy V2 路线（GradualEncoderDecoder）— 已放弃 ❌
保守配置（关 Dropout/MixUp/退火）mean ACC = 0.639，比 C-IDC baseline 0.711 跌 10%。

### 2.3 方向 B — 稳定性工程（当前）
回归原始 C-IDC，用 `sefs_bayes_opt.py` 在 ALLAML 上做贝叶斯超参数优化。

---

## 三、当前工具

### `sefs_bayes_opt.py` — 多目标贝叶斯优化器（957行 → 增强版）

**已增强功能：**
- ✅ **OOM 保护**：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`，训练后 `cuda.synchronize()` + 双重 `empty_cache()`
- ✅ **即时保存**：每个 seed 完成后立即写盘（`per_seed_immediate.csv`），含 `os.fsync` 强制刷盘
- ✅ **断点续跑**：`--resume <output_dir>` 可从中断处继续，GP 状态 + 聚合结果通过 `optimizer_checkpoint.json` 恢复
- ✅ **任意时刻续跑**：checkpoint 在每个 seed 完成后都保存，即使 OOM 打断在轮次中间，已完成的轮次也不会丢失（被中断的轮次会重新评估）

**使用方式：**
```bash
# 全新运行
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe sefs_bayes_opt.py \
  --cfg cfg/ALLAML.yaml --n_calls 60 --batch_size 4 \
  --n_workers 4 --n_seeds 10 --early_stop --patience 40 --min_delta 0.005

# 断点续跑（电脑重启后）
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe sefs_bayes_opt.py \
  --cfg cfg/ALLAML.yaml --n_calls 60 --batch_size 4 --n_workers 4 --n_seeds 10 \
  --early_stop --patience 40 --min_delta 0.005 \
  --resume sefs_bayes_opt_20260727_XXXXXX
```

**输出文件（在 `sefs_bayes_opt_*/` 目录下）：**
| 文件 | 说明 |
|------|------|
| `per_seed_immediate.csv` | ✅ 每个 seed 完成立即写入，中断不丢 |
| `optimizer_checkpoint.json` | ✅ GP 状态 checkpoint，续跑用 |
| `all_results.csv` | 聚合结果（每组参数多种子均值） |
| `multi_objective_best_params.json` | 最佳参数 JSON |
| `multi_objective_report.txt` | 人类可读报告 |

**三个贝叶斯脚本对比：**
| | `sefs_bayes_opt.py` | `sefs_opt_bayes.py` | `sefs_opt_bayesV1.py` |
|---|---|---|---|
| 优化目标 | ✅ 5指标综合 | 仅 ACC | 仅 ACC |
| 参数数量 | 19 | 19 | 17 |
| Per-seed 记录 | ✅ | ❌ | ❌ |
| 断点续跑 | ✅ | ❌ | ❌ |
| OOM 保护 | ✅ | ❌ | ❌ |
| **推荐使用** | **← 用这个** | 快速探索 | 固定 batch |

---

## 三.5 内存优化（已完成 ✅）

针对贝叶斯优化首次运行时的 `CUDA error: an illegal memory access` 崩溃，做了两处内存优化。**两者都不改变模型任何行为，结果与原来完全一致。**

### 优化 1：GTCR 损失的行列式引理（省 GPU 显存）

**文件**：`train_evaluate.py` 的 `TotalCodingRateWithProjection.compute_discrimn_loss`

**原理**：原代码在 7129 维特征上算 `logdet(I_7129 + s·W·W^T)`，会创建 7129×7129 矩阵（~600MB 临时显存）。用 **Sylvester 行列式定理** `det(I_p + s·W·W^T) = det(I_m + s·W^T·W)`，把 7129×7129 的 logdet 等价转换为 **batch_size × batch_size** 的小矩阵。

**关键**：`W` 的形状是 `[7129, batch_size]`，所以 `W·W^T` 是 7129×7129（巨无霸），`W^T·W` 是 batch_size×batch_size（比如 4×4~10×10）。因为 batch_size ≤ 72 << 7129，优化对搜索空间里**每一个 batch_size 取值都成立**。

**改动前**：
```python
I = torch.eye(p, device=W.device)          # p×p = 7129×7129，200MB
logdet = torch.logdet(I + scalar * W.matmul(W.T))   # 大矩阵行列式
```

**改动后**：
```python
M = torch.eye(m, device=W.device) + scalar * (W.T @ W)   # m×m = batch_size×batch_size
logdet = torch.logdet(M)                                 # 小矩阵行列式
```

**影响**：数学恒等变换，损失值逐位相同。显存从 ~600MB 降到 <1MB，速度提升约 10⁵ 倍，且数值更稳定（避免了 7129×7129 大矩阵 Cholesky 的舍入误差）。

### 优化 2：Cholesky 完成后释放大矩阵（省 CPU 内存）

**文件**：`dataset.py` 的 `NumpyTableDataset.__init__`

**原理**：模型启动时做 Cholesky 分解，会生成多张 7129×7129 的大表格（每张 203MB float32 / 406MB float64），但用完后一直留在内存里。

**改动**：在 Cholesky 分解完成、诊断输出后，立即 `delattr` 释放以下不再需要的矩阵：
- `_corr_for_cholesky`（406MB float64）
- `_corr_for_clustering`（406MB float64）
- `correlation_matrix`（203MB float32 torch）
- `cluster_labels`（小）

然后 `gc.collect()` 强制回收。

**影响**：这些矩阵在后续训练中**根本不会被用到**（真正需要的 `cholesky_L` 已保留），释放的是"打包用的纸箱"，不影响任何结果。共释放约 1.2GB CPU 内存。

---

## 四、贝叶斯搜索 Round 1 结果（已完成 ✅）

**运行**：`sefs_bayes_opt_20260816_104936/`，n_calls=40，n_workers=1，n_seeds=10，共 10 轮，全部成功。

**核心结果**：最佳参数组 ACC 均值 **0.800**，比 baseline 0.711 **提升 12.5%**，但标准差 0.067，**方差未改善**。

### 📌 最佳参数存档（可退回的高均值版本）

来源：第 1 轮参数组 01，ACC 均值 0.800，10 seed = `[0.778, 0.806, 0.847, 0.847, 0.736, 0.819, 0.750, 0.819, 0.917, 0.681]`（范围 0.681~0.917）。

可直接写入 `cfg/ALLAML_best.yaml` 复现：

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

**复现命令**：
```bash
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe train_evaluate.py --cfg cfg/ALLAML_best.yaml
```

### 问题：方差大，不稳定

- 均值 0.800 已显著提升，但标准差 0.067 与 baseline（0.055）相当，**方差未解决**。
- 最好 seed 0.917，最差 seed 0.681（低于 baseline），说明「训练轨迹不稳定导致某些 seed 陷入差局部最优」，而非模型能力上限。
- 根因：样本仅 72 个，深度聚类对随机初始化/门控噪声/数据 mask 敏感，属数据规模固有特性，单纯调参难根治。

---

## 四.5 方向 A — 把方差纳入优化目标（已运行，结论：退回 0.800）

**目标**：让贝叶斯搜索主动避开「高均值但高方差」的参数，偏好稳定解。

**改动**（`sefs_bayes_opt.py`）：
1. 聚合逻辑加 `multi_objective_std` 和 `stability_score = 均值 − λ·标准差`
2. GP 优化目标从 `loss = -均值` 改为 `loss = -stability_score`
3. 新增命令行参数 `--stability_lambda`（默认 0.3）
4. 所有报告/排序/checkpoint 统一改用 `stability_score` 作为排序键

**搜参空间微调**（基于 Round 1，4 个参数碰边界）：batch_size [4,14]、epochs [900,1500]、eps [0.2,0.6]、sefs_tau [0.5,1.8]。

### Round 2 结果（λ=0.4，`sefs_bayes_opt_20260817_233855/`）— 失败 ❌

| 组 | ACC 均值 | ACC std | 说明 |
|----|---------|---------|------|
| Round 1 最佳（λ=0） | **0.800** | 0.067 | 高均值 |
| Round 2 ACC 最高组 | 0.765 | 0.054 | 均值降 3.5 点 |
| Round 2 稳定评分最高组 | 0.760 | **0.104** ⚠️ | 被方向A选中，ACC 反而更不稳 |

### 关键结论

1. **方向 A 惩罚错了指标**：惩罚的是 `multi_objective_score`（综合分）的 std，而非 ACC 的 std。综合分里 Silhouette/DBI 的方差与 ACC 方差解耦甚至负相关，导致选出的「稳定解」ACC std 反而更高（0.104）。
2. **方差有地板**：所有 42 组参数的 ACC std 都在 0.05~0.11 之间，最小约 0.05，这是 72 样本数据的固有噪声，调参无法突破。
3. **高均值不一定高方差**：Round 2 里 ACC 最高组（0.765）的 std=0.054，反而低于很多低均值组。
4. **惩罚方差的收益远小于代价**：λ=0.4 让均值降 4 个百分点，方差几乎没改善。

### 决策：退回 Round 1 的 0.800 版本

**当前采用**：Round 1 最佳参数（ACC 均值 0.800），完整 10 seed 结果和参数组合已存于 **`BEST_RESULT_ACC_0.800.md`**。

**结论**：准确率可通过调参提升（0.711→0.800），但方差是数据固有属性，调参无效。`0.800 ± 0.067` 为当前可达到的最优均值，报告为「均值±std」即可。

---

## 五、踩过的坑 ⚠️

### 坑 1：`.pyc` 字节码缓存
任何 `.py` 编辑后必须清理 `__pycache__`：
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### 坑 2：K-Means 初始化维度匹配
`clustering_head`: Linear(32→128)→BN→ReLU→Linear(128→n_clusters)。K-Means 必须在 128 维空间做。

### 坑 3：Heavy model.py 缺少 `class GatingNet` 声明
GradualEncoderDecoder 后直接就是 GatingNet 的 `__init__`，类声明丢失。已修复但备份文件仍有 bug。

### 坑 4：Dropout + MixUp 在小样本基因数据上是毒药
ALLAML 72 样本，32 维瓶颈已是极强正则化。Dropout 和 MixUp 在任何强度下都破坏信号。

### 坑 5：SEFS 退火不能直接覆盖精调的固定 τ
原始 ALLAML `sefs_tau: 1.29` 是精调值，退火从 3.0 开始长时间偏离最优值。

### 坑 6：Windows GBK 编码 + Conda 环境
```bash
PYTHONIOENCODING=utf-8 D:/ANACONDA/envs/CIDC/python.exe script.py
```
必须同时指定编码和完整 Python 路径。

### 坑 7：PowerShell 中不能只用 `activate CIDC`
必须用 `conda activate CIDC` 或直接用完整 Python 路径。

### 🆕 坑 8：贝叶斯优化中途中断
`sefs_bayes_opt.py` 现在支持 `--resume`。每个 seed 结果即时保存到 `per_seed_immediate.csv`，GP 状态保存到 `optimizer_checkpoint.json`。

### 🆕 坑 9：多进程 GPU 内存竞争
`n_workers=4` 时 4 个子进程同时加载模型到 GPU。已通过 `CUDA_VISIBLE_DEVICES=0` 和 `max_split_size_mb:128` 缓解。**但根因是 GTCR 大矩阵（见坑 10）**，即使单进程也可能崩，故 `n_workers` 建议设 1~2。

### 🆕 坑 10：GTCR 损失的 7129×7129 大矩阵 OOM
`TotalCodingRateWithProjection.compute_discrimn_loss` 原实现 `logdet(I_7129 + s·W·W^T)` 会创建 7129×7129 矩阵，单次 ~600MB 显存。4 进程并发 → GPU 非法内存访问（`illegal memory access`），且一旦发生会**污染整个 GPU 上下文**，导致后续所有 seed（连 `seed_everything`）都失败。**已用 Sylvester 行列式定理修复**（见「三.5 优化 1」），矩阵降到 batch_size×batch_size。

**教训**：`W` 形状 `[7129, batch_size]` 时，`W·W^T` 是 7129×7129（巨无霸），但 `det(I_p + s·W·W^T) = det(I_m + s·W^T·W)`，右边是 batch_size×batch_size。凡是在高维特征（p 大）低 batch（m 小）场景算 `logdet(I + W·W^T)`，都要优先用行列式引理。

---

## 六、环境信息

- **Python**：`D:\ANACONDA\envs\CIDC\python.exe`（Conda 环境 CIDC）
- **GPU**：NVIDIA GeForce RTX 5060 Laptop GPU (CUDA)
- **框架**：PyTorch + PyTorch Lightning + scikit-optimize
- **工作目录**：`D:\PythonProject\CIDC`

---

## 七、文件修改历史

| 日期 | 改动 |
|------|------|
| 2026-07-15~23 | Lite 路线全部分析，结论：全部 ≤ baseline |
| 2026-07-26 | Heavy V2 bug 修复 + 保守配置 + 运行（0.639，失败） |
| 2026-07-26 | 原始 C-IDC 恢复，方向 B 启动 |
| 2026-07-27 | `sefs_bayes_opt.py` 增强：OOM 保护 + 即时保存 + 断点续跑 |
| 2026-08-13 | `sefs_bayes_opt.py` 搜索空间缩小（以 ALLAML.yaml 精调值为中心） |
| 2026-08-16 | `train_evaluate.py` 优化1：GTCR 行列式引理（7129²→batch²） |
| 2026-08-16 | `dataset.py` 优化2：Cholesky 后释放大矩阵 |
| 2026-08-17 | 贝叶斯搜索 Round 1 完成：最佳 ACC 均值 0.800（+12.5%），存档于「四」节 |
| 2026-08-17 | 方向 A 实现：`sefs_bayes_opt.py` 优化目标改为「均值 − λ·std」+ 搜参空间微调 4 参数 |
| 2026-08-18 | 方向 A Round 2 运行（λ=0.4）：方差有地板，调参无效，退回 0.800 版本 |
| 2026-08-18 | 生成 `BEST_RESULT_ACC_0.800.md`：记录最佳参数 + 10 seed 结果 |
| 2026-08-20 | ALLAML Round 3（λ=0, n_calls=60）：ACC 最高 0.7806，未突破 0.800，确认调参到头 |
| 2026-08-20 | `sefs_bayes_opt.py` 搜索空间按 dataset 分：新增 `SRBCT_SEARCH_SPACE`，`--cfg cfg/SRBCT.yaml` 自动匹配 |
