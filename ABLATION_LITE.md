# C-IDC 轻量改进消融实验设计

## 双层消融结构

### 层A：C-IDC vs IDC 组件消融（不改动代码，仅调配置）
### 层B：轻量改进消融（D1-D6）

---

# 层A：C-IDC vs IDC 组件消融

## 实验目标
量化 C-IDC 的 SEFS Gaussian Copula 链路中每个数学步骤的贡献。

## C-IDC SEFS 链路拆解

```
IDC 独立噪声
  └─ [A1] 相关矩阵 R = OAS估计
       └─ [A2] Cholesky 分解 L (R = LL^T)
            └─ [A3] 相关噪声 v = L @ ε
                 └─ [A4] Gaussian CDF u = Φ(v)
                      └─ [A5] reparam m̃ = σ((logit(π)+logit(u))/τ)
                           └─ C-IDC 完整链路
```

## 层A实验矩阵

| 实验ID | 配置 | 验证的组件 |
|--------|------|-----------|
| A0_IDC | `use_sefs: false` | IDC 原版独立噪声（下界） |
| A1 | `use_sefs: true`, cholesky_L=Identity | 仅相关矩阵（OAS）+ 单位Cholesky |
| A2 | 用 `_compute_cholesky_L`(Full) 替代 Block Cholesky | Block vs Full Cholesky |
| A3 | A1 + Φ映射 + reparam | 无相关结构但有Copula变换 |
| A4_CIDC | `use_sefs: true`（当前默认） | C-IDC 完整链路（上界） |
| A5 | A4 + sefs_tau=0.5 | 固定低温度（硬选择）|
| A6 | A4 + sefs_tau=3.0 | 固定高温度（软选择）|

## 层A运行方式

```bash
# A0: IDC 原版
# 修改配置文件: use_sefs: false
python train_evaluate.py --cfg cfg/ALLAML.yaml

# A4: C-IDC 当前默认（use_sefs: true，其余同原始配置）
python train_evaluate.py --cfg cfg/ALLAML.yaml

# A1/A2/A3: 需要少量代码修改（注释掉对应步骤）
# 在 model.py GatingNet.forward 中：
#   A1: 注释掉 torch.matmul(epsilon, self.cholesky_L.t())，用 v=epsilon
#   A2: 修改 dataset.py 中的 cholesky 计算方法
#   A3: 注释掉 Φ 映射那一行，直接用 v
```

---

# 层B：轻量改进消融（D1-D6）

## 6个轻量改进

| ID | 改进 | 配置字段 | 改动行数 |
|----|------|---------|---------|
| D1 | EncoderDecoder + Dropout | `encdec_dropout: 0.2` | model.py ~10行 |
| D2 | AdamW 替代 Adam | `weight_decay_pretrain/cluster` | train_evaluate.py ~5行 |
| D3 | SEFS 温度退火 | `use_sefs_annealing: true` | model.py ~25行 + train_evaluate.py ~5行 |
| D4 | 梯度裁剪 | `grad_clip_norm: 1.0` | train_evaluate.py ~8行 |
| D5 | K-Means 聚类头初始化 | `use_kmeans_init: true` | train_evaluate.py ~40行 |
| D6 | Tabular MixUp | `mixup_alpha: 0.2` | train_evaluate.py ~15行 |
| D7 | 集群平衡损失 | `balance_loss_lambda: 0.1` | train_evaluate.py ~5行 |

## 层B实验矩阵

以 C-IDC 当前默认配置为 Baseline（use_sefs: true, 无 D1-D7）。

| 实验ID | 启用的改进 | 配置差异 |
|--------|-----------|---------|
| B0 | Baseline（C-IDC 默认） | 所有 D1-D7 关闭 |
| B1 | D1 only | `encdec_dropout: 0.2` |
| B2 | D2 only | `weight_decay_pretrain: 1e-4` |
| B3 | D3 only | `use_sefs_annealing: true` |
| B4 | D4 only | `grad_clip_norm: 1.0` |
| B5 | D5 only | `use_kmeans_init: true` |
| B6 | D6 only | `mixup_alpha: 0.2` |
| B7 | D7 only | `balance_loss_lambda: 0.1` |
| B8 | D1+D3（正则化组合） | Dropout + SEFS退火 |
| B9 | D2+D4+D7（训练组合） | AdamW + GradClip + Balance |
| B10 | D5+D6（数据+初始化组合） | K-Means + MixUp |
| B11 | Full Lite（D1+D2+D3+D4+D5+D6+D7） | 全部启用 |

## 层B一键运行

```bash
# 生成层B消融配置
python ablation_lite.py --mode generate

# 运行全部层B实验（11个实验 × 3个数据集 × 10 seeds = 330次训练）
# 建议分数据集并行跑
python ablation_lite.py --mode run --dataset ALLAML
python ablation_lite.py --mode run --dataset Prostate
python ablation_lite.py --mode run --dataset SRBCT

# 汇总结果
python ablation_lite.py --mode summarize
```

## 分析Prompt

将以下内容连同结果表一起提交给AI分析：

```markdown
# C-IDC 轻量改进消融分析

## 实验数据
[粘贴B0-B11的ACC/ARI/NMI表格]

## 分析问题

### 1. 层A（C-IDC vs IDC）
1. 相比 IDC 独立噪声（A0），C-IDC 完整链路（A4）在各数据集上的增益是多少？
2. Gaussian Copula 的哪个步骤贡献最大？（对比 A1 vs A3 vs A4）
3. sefs_tau 固定值（A5=0.5, A6=3.0）vs 退火（D3）哪个更好？

### 2. 层B（轻量改进贡献排序）
1. 对 D1-D7 按 ACC 贡献从大到小排序
2. 是否有改进在某个数据集上为负贡献？
3. B8（D1+D3）的组合效果是否大于单独之和（正交互）？
4. B9（训练组合）和 B10（数据/初始化组合）哪个贡献更大？

### 3. 数据集特异性
1. SRBCT（63样本）是否对 D1（Dropout）和 D6（MixUp）更敏感？
2. ALLAML（7129特征）是否对 D3（SEFS退火）和 D5（K-Means）更敏感？

### 4. 最终推荐
1. 对每个数据集，哪些改进是必须保留的（贡献 > +2%）？
2. 是否存在可以移除的改进（贡献 < 0）？
3. 给出三个数据集的最简配置（仅保留必要改进）。

请基于数据给出分析，不做无根据的推断。
```
