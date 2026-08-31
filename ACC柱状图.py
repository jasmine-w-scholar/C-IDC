import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------- 数据 ----------------
datasets = ['ALLAML', 'SRBCT', 'PROSTATE']
methods  = ['C-IDC', 'k-DVAE', 'IDC', 'KM', 'TELL']

# 行 = 方法，列 = 数据集
values = np.array([
    [80.0, 56.8, 68.2],   # C-IDC  (ALLAML 已由 78.9 改为 80.0)
    [72.6, 49.4, 59.5],   # k-DVAE
    [77.5, 55.4, 65.3],   # IDC
    [67.3, 39.6, 58.1],   # KM
    [66.7, 38.9, 63.6],   # TELL
])

colors = ['#3B5F87',   # 深蓝
          '#B92B2B',   # 砖红
          '#8DB3C7',   # 浅蓝
          '#E5A0A0',   # 粉红
          '#B3B3AE']   # 灰

# ---------------- 全局样式 ----------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 1.2,
})

fig, ax = plt.subplots(figsize=(10.2, 6.4), dpi=150)

n_m = len(methods)
bar_w = 0.155                      # 单根柱宽
group_gap = 0.30                   # 组间空隙(用于计算组中心)
x = np.arange(len(datasets)) * (n_m * bar_w + group_gap)

for i, (m, c) in enumerate(zip(methods, colors)):
    pos = x + i * bar_w
    ax.bar(pos, values[i], width=bar_w, color=c,
           edgecolor='black', linewidth=0.9, label=m, zorder=3)
    # 数值标签
    for xi, v in zip(pos, values[i]):
        ax.text(xi, v + 0.5, f'{v:.1f}%', ha='center', va='bottom',
                fontsize=9.5, zorder=4)

# ---------------- 坐标轴 ----------------
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_ylim(30, 85)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.set_xticks(x + bar_w * (n_m - 1) / 2)
ax.set_xticklabels(datasets, fontsize=13, fontweight='bold')
ax.set_xlim(x[0] - bar_w, x[-1] + n_m * bar_w)

ax.tick_params(axis='y', labelsize=12, length=4)
ax.tick_params(axis='x', length=0)

# 虚线水平网格
ax.grid(axis='y', linestyle='--', color='#BFBFBF', linewidth=0.9, zorder=0)
ax.set_axisbelow(True)

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

# ---------------- 图例（右上、两列） ----------------
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.02),
          ncol=2, frameon=False, fontsize=11.5,
          handlelength=1.4, handleheight=1.1,
          columnspacing=1.6, labelspacing=0.45)

plt.tight_layout()
plt.savefig('ACC.png', dpi=300, bbox_inches='tight')
plt.savefig('ACC.pdf', bbox_inches='tight')
plt.show()