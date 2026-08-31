# C-IDC 消融实验设计方案

## 实验目标

量化以下 5 个改进各自对三个数据集（ALLAML、PROSTATE、SRBCT）聚类性能的贡献：

| 改进编号 | 改进名称 | 关键配置字段 |
|----------|----------|-------------|
| I1 | 渐进压缩 Encoder-Decoder | `use_gradual_encdec`, `encdec_div_factor`, `encdec_dropout` |
| I2 | SEFS 温度退火 | `use_sefs_annealing`, `sefs_tau_start`, `sefs_tau_end` |
| I3 | K-Means 聚类头初始化 | `use_kmeans_init` |
| I4 | AdamW + Warmup + 梯度裁剪 + 平衡损失 | `weight_decay_*`, `warmup_ratio`, `grad_clip_norm`, `balance_loss_lambda` |
| I5 | Tabular MixUp 增强 | `mixup_alpha` |

---

## 实验矩阵

### Baseline（无任何改进）
```
use_gradual_encdec: false     # 原始 EncoderDecoder
use_sefs_annealing: false     # 固定 sefs_tau
use_kmeans_init: false        # 随机初始化
optimizer: Adam               # 非 AdamW
warmup_ratio: 0               # 无 warmup
grad_clip_norm: 0             # 无裁剪
balance_loss_lambda: 0        # 无平衡损失
mixup_alpha: 0                # 无 MixUp
```

### 实验 1-5：单改进消融（每次只启用一个改进）

| 实验 | 启用的改进 | 预期增益来源 |
|------|-----------|-------------|
| E1 | 仅 I1（渐进压缩） | 架构正则化：参数减少 + Dropout + LayerNorm |
| E2 | 仅 I2（SEFS 退火） | 特征选择质量：从软到硬的渐进门控 |
| E3 | 仅 I3（K-Means 初始化） | 更好的聚类初始点 |
| E4 | 仅 I4（训练增强） | AdamW 正则化 + Warmup 稳定性 + 平衡集群 |
| E5 | 仅 I5（MixUp） | 数据增强：增广小样本训练空间 |

### 实验 6-10：组合消融（逐步叠加）

| 实验 | 累积改进 | 目的 |
|------|---------|------|
| E6 | Baseline + I1 + I2 | 架构 + 特征选择 |
| E7 | E6 + I3 | 加上聚类初始化 |
| E8 | E7 + I4 | 加上训练增强 |
| E9 | E8 + I5 | 全部 5 个改进（Full Model） |
| E10 | Full - I1 | 验证 I1（架构）的必要性 |

---

## 消融实验配置生成器

以下是生成各实验配置文件的 Python 脚本。每个实验在三个数据集上各运行 10 seed。

### 运行命令模板

```bash
# 单实验运行（以 ALLAML + Full Model 为例）
python train_evaluate.py --cfg cfg/ablation/ALLAML_full.yaml

# 批量运行所有消融实验
python ablation_study.py --dataset ALLAML --seeds 10
python ablation_study.py --dataset Prostate --seeds 10
python ablation_study.py --dataset SRBCT --seeds 10
```

---

## 消融实验 Prompt（供 Code Review / 分析用）

将以下 Prompt 与消融实验结果一起提供给 AI，可获得结构化的分析报告：

```markdown
# 消融实验分析 Prompt

请根据以下消融实验结果，分析 C-IDC 各改进的贡献：

## 实验数据

[粘贴消融实验结果表，包含每个实验在三个数据集上的 ACC/ARI/NMI 均值±标准差]

## 分析要求

### 1. 单改进贡献分析
对每个改进（I1-I5），分别计算在三个数据集上的**平均 ACC 增益**（相对于 Baseline）。
排序找出最有效和最低效的改进。

### 2. 改进间交互效应
- 是否存在**正交互**（组合效果 > 单独效果之和）？
- 是否存在**负交互**（组合效果 < 单独效果之和）？
- 特别关注 I2（SEFS 退火）和 I5（MixUp）是否冲突（退火倾向于硬选择，MixUp 引入软混合）

### 3. 数据集特异性
- 哪个数据集对改进最敏感（增益最大）？为什么？（考虑样本量、特征数、类别数）
- 哪个改进在三个数据集上表现最一致（方差最小）？

### 4. 架构改进 vs 训练改进
- I1（渐进压缩架构）和 I4（训练增强）哪个贡献更大？
- 是否存在"I1 改进后 I4 的贡献变小"的现象？（即架构正则化后训练增强的边际收益递减）

### 5. 结论与建议
- 哪些改进是**必须保留**的（贡献 > +2%）？
- 哪些改进是**可选**的（贡献 < +1% 但不伤害性能）？
- 是否存在需要**回退**的改进（贡献为负）？
- 对三个数据集分别给出**最优配置推荐**。

请给出基于数据的分析，不要做无根据的推断。
```

---

## 结果记录表模板

| 实验 | ALLAML ACC | ALLAML ARI | ALLAML NMI | PROSTATE ACC | PROSTATE ARI | PROSTATE NMI | SRBCT ACC | SRBCT ARI | SRBCT NMI |
|------|-----------|-----------|-----------|-------------|-------------|-------------|----------|----------|----------|
| Baseline | | | | | | | | | |
| E1 (I1) | | | | | | | | | |
| E2 (I2) | | | | | | | | | |
| E3 (I3) | | | | | | | | | |
| E4 (I4) | | | | | | | | | |
| E5 (I5) | | | | | | | | | |
| E6 (I1+I2) | | | | | | | | | |
| E7 (I1+I2+I3) | | | | | | | | | |
| E8 (I1+I2+I3+I4) | | | | | | | | | |
| E9 (Full) | | | | | | | | | |
| E10 (Full-I1) | | | | | | | | | |

---

## 统计检验建议

1. **配对 t 检验**：将同一个 seed 下的 Baseline vs Full Model 配对，计算 p 值
2. **效应量**：Cohen's d = (μ_full - μ_baseline) / σ_pooled
3. **胜率**：在 10 个 seed 中 Full Model ACC > Baseline ACC 的比例
