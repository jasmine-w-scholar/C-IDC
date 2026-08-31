import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['pdf.fonttype'] = 42   # 确保 PDF 中字体可编辑（Type 3 → Type 42）
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'DejaVu Sans'

data = {
    'Prostate': {
        'tau':      [0.3,  0.5,  0.8,  1.0,  1.2,  1.3,  1.4,  1.5,  1.8,  2.0,  3.0],
        'acc_mean': [0.680392, 0.649020, 0.643137, 0.635294, 0.680392,
                     0.639216, 0.625490, 0.629412, 0.645098, 0.627451, 0.664706],
        'acc_std':  [0.047059, 0.063051, 0.051356, 0.021834, 0.066954,
                     0.043581, 0.015686, 0.011433, 0.069987, 0.019608, 0.068879],
    },
    'AMLALL': {
        'tau':      [0.3,  0.5,  0.8,  1.0,  1.2,  1.3,  1.4,  1.5,  1.8,  2.0,  3.0],
        'acc_mean': [0.761111, 0.752778, 0.786111, 0.761111, 0.758333,
                     0.744444, 0.738889, 0.744444, 0.688889, 0.725000, 0.641667],
        'acc_std':  [0.063586, 0.053720, 0.046148, 0.098758, 0.054575,
                     0.077877, 0.077778, 0.073283, 0.104157, 0.072115, 0.153759],
    },
    'SRBCT': {
        'tau':      [0.3,  0.5,  0.8,  1.0,  1.2,  1.3,  1.4,  1.5,  1.8,  2.0,  3.0],
        'acc_mean': [0.520482, 0.513253, 0.503614, 0.436145, 0.443373,
                     0.443373, 0.433735, 0.390361, 0.409639, 0.390361, 0.424096],
        'acc_std':  [0.023362, 0.039300, 0.083958, 0.038403, 0.045337,
                     0.026833, 0.086881, 0.055787, 0.064657, 0.044169, 0.069588],
    },
}
# ── 用户指定配色 ──────────────────────────────────────────────
COLORS = {
    'Prostate': '#CA0E12',   # 深蓝灰
    'AMLALL':   '#013E75',   # 珊瑚红
    'SRBCT':    '#fdb338',   # 暖橙
}

# ── 标记符号 ─────────────────────────────────────────────────
MARKERS = {
    'Prostate': 's',    # 正方形
    'AMLALL':   '^',    # 上三角
    'SRBCT':    'D',    # 菱形
}

# ── X 轴错位偏移（三条线左中右错开，单位与 tau 一致）──────────
# 让误差棒在视觉上分离，不重叠
JITTER = {
    'Prostate': -0.045,
    'AMLALL':    0.000,
    'SRBCT':    +0.045,
}
# ── 标准差上限截断（clamp）─────────────────────────────────────
STD_CAP = 0.048
# ============================================================
# Cell 4 — 核心绘图逻辑
# ============================================================

fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=150)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── 逐数据集绘制带截断误差棒的折线 ──────────────────────────
for name in ['Prostate', 'AMLALL', 'SRBCT']:
    d        = data[name]
    taus     = np.array(d['tau'])
    means    = np.array(d['acc_mean'])
    raw_stds = np.array(d['acc_std'])

    # 截断过大的标准差，避免误差棒视觉上"充电"
    clipped_stds = np.clip(raw_stds, 0.0, STD_CAP)

    # X 轴错位（jitter），三条线之间留出间隙
    taus_jittered = taus + JITTER[name]

    ax.errorbar(
        taus_jittered,
        means,
        yerr=clipped_stds,
        label=name,
        color=COLORS[name],
        marker=MARKERS[name],
        markersize=6.5,
        linewidth=1.9,
        capsize=3.8,
        capthick=1.3,
        elinewidth=1.1,
        zorder=3,
    )

# ────────────────────────────────────────────────────────────
# 横轴刻度设计
# 原始 tau 值：[0.3, 0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 3.0]
# 问题：1.2/1.3/1.4/1.5 四个值间距很小（Δ=0.1），紧密簇拥
# 解决方案：
#   ① 通过 set_xlim 在首尾增加留白，整体视觉更均匀
#   ② 适当减小 labelsize，避免数字碰撞
#   ③ tick 文字格式化：整数不加小数点，小数保留最短形式
# ────────────────────────────────────────────────────────────
tau_ticks = [0.3, 0.5, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 3.0]
ax.set_xticks(tau_ticks)

def _fmt_tau(x, _):
    """将浮点刻度格式化为紧凑字符串：1.0→'1', 0.3→'0.3'"""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    # 去掉尾随零（如 0.50 → 0.5）
    s = f'{x:.2f}'.rstrip('0').rstrip('.')
    return s

ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tau))

# 首尾留白让刻度分布看起来更均匀
ax.set_xlim(0.12, 3.28)

# 横轴数字字号调小，防止拥挤（尤其 1.2/1.3/1.4/1.5 密集区）
ax.tick_params(axis='x', labelsize=9.0, pad=4)
ax.tick_params(axis='y', labelsize=10.5)

# 纵轴范围
ax.set_ylim(0.32, 0.88)

# ── 轴标签：横轴改为 \tau（无下标）──────────────────────────
ax.set_xlabel(r'$\tau$', fontsize=15, labelpad=6)
ax.set_ylabel('ACC',     fontsize=13, labelpad=6)

# ── 图例 ─────────────────────────────────────────────────────
ax.legend(
    fontsize=10.5,
    loc='upper right',
    framealpha=0.93,
    edgecolor='#cccccc',
    handlelength=2.2,
    borderpad=0.7,
)

# ── 去除上/右边框（简洁学术风格）────────────────────────────
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# ── 横向辅助网格（淡色，不抢主体）──────────────────────────
ax.yaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.45, color='#999999', zorder=0)
ax.set_axisbelow(True)

plt.tight_layout(pad=1.3)
plt.show()
OUTPUT_PNG = 'sensitivity_tau_acc.png'
OUTPUT_PDF = 'sensitivity_tau_acc.pdf'

# ── 重新绘制（确保保存的与显示的完全一致）──────────────────
fig2, ax2 = plt.subplots(figsize=(8.8, 4.6))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

for name in ['Prostate', 'AMLALL', 'SRBCT']:
    d        = data[name]
    taus     = np.array(d['tau'])
    means    = np.array(d['acc_mean'])
    raw_stds = np.array(d['acc_std'])
    clipped_stds   = np.clip(raw_stds, 0.0, STD_CAP)
    taus_jittered  = taus + JITTER[name]

    ax2.errorbar(
        taus_jittered, means, yerr=clipped_stds,
        label=name,
        color=COLORS[name],
        marker=MARKERS[name],
        markersize=6.5,
        linewidth=1.9,
        capsize=3.8,
        capthick=1.3,
        elinewidth=1.1,
        zorder=3,
    )

ax2.set_xticks(tau_ticks)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_tau))
ax2.set_xlim(0.12, 3.28)
ax2.set_ylim(0.32, 0.88)
ax2.tick_params(axis='x', labelsize=9.0, pad=4)
ax2.tick_params(axis='y', labelsize=10.5)
ax2.set_xlabel(r'$\tau$', fontsize=15, labelpad=6)
ax2.set_ylabel('ACC',     fontsize=13, labelpad=6)
ax2.legend(fontsize=10.5, loc='upper right',
           framealpha=0.93, edgecolor='#cccccc',
           handlelength=2.2, borderpad=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.8)
ax2.spines['bottom'].set_linewidth(0.8)
ax2.yaxis.grid(True, linestyle=':', linewidth=0.7,
               alpha=0.45, color='#999999', zorder=0)
ax2.set_axisbelow(True)
plt.tight_layout(pad=1.3)

# ── 保存 ─────────────────────────────────────────────────────
fig2.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
fig2.savefig(OUTPUT_PDF,           bbox_inches='tight', facecolor='white')
plt.close(fig2)

print(f"✅ 图片已保存：")
print(f"   PNG  →  {OUTPUT_PNG}  (300 dpi)")
print(f"   PDF  →  {OUTPUT_PDF}  (矢量图，适合论文投稿)")