#!/usr/bin/env python
"""
ALLAML SEFS 消融实验 t 检验。

对比 use_sefs=true（完整 C-IDC）vs use_sefs=false（w/o SEFS）的 10 seed ACC，
做独立样本 Welch's t 检验 + 配对 t 检验 + Cohen's d 效应量。

用法:
    python ablation_ttest.py
"""
import numpy as np
from scipy import stats

# ── 消融实验数据（各 10 seed 的 ACC）──
# use_sefs=true（完整 C-IDC，最优超参数）
acc_true = [
    0.7778, 0.8056, 0.8472, 0.8472, 0.7361,
    0.8194, 0.7500, 0.8194, 0.9167, 0.6806,
]
# use_sefs=false（w/o SEFS，退化为 IDC）
acc_false = [
    0.6388888888888888, 0.75, 0.75, 0.8611111111111112, 0.7777777777777778,
    0.8055555555555556, 0.9305555555555556, 0.7638888888888888, 0.75, 0.7361111111111112,
]

mean_t = np.mean(acc_true)
mean_f = np.mean(acc_false)
std_t = np.std(acc_true, ddof=1)
std_f = np.std(acc_false, ddof=1)

# ── 独立样本 t 检验（Welch's t-test，不假设等方差）──
t_ind, p_ind = stats.ttest_ind(acc_true, acc_false, equal_var=False)

# ── 配对 t 检验（相同 seed 对齐）──
t_rel, p_rel = stats.ttest_rel(acc_true, acc_false)

# ── Cohen's d 效应量 ──
pooled_std = np.sqrt((std_t ** 2 + std_f ** 2) / 2)
cohen_d = (mean_t - mean_f) / pooled_std

print("=" * 60)
print("ALLAML SEFS 消融实验 t 检验")
print("=" * 60)
print(f"use_sefs=true  (C-IDC): 均值={mean_t:.4f}, std={std_t:.4f}")
print(f"use_sefs=false (≈IDC) : 均值={mean_f:.4f}, std={std_f:.4f}")
print(f"均值差                 : {mean_t - mean_f:.4f} "
      f"({(mean_t - mean_f) / mean_f * 100:.2f}% 相对提升)")
print()
print(f"[独立样本 Welch's t 检验] t={t_ind:.4f}, p={p_ind:.4f}")
print(f"[配对 t 检验]             t={t_rel:.4f}, p={p_rel:.4f}")
print(f"[Cohen's d 效应量]        d={cohen_d:.4f}")
print()
print("显著性判断（α=0.05）:")
print(f"  独立样本检验: {'显著 ✅' if p_ind < 0.05 else '不显著 ⚠️'} (p={p_ind:.4f})")
print(f"  配对检验    : {'显著 ✅' if p_rel < 0.05 else '不显著 ⚠️'} (p={p_rel:.4f})")
print()
print("Cohen's d 解释: 0.2=小, 0.5=中, 0.8=大")
