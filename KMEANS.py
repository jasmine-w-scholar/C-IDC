import platform

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    confusion_matrix
)
from munkres import Munkres
import platform
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Windows/Linux
matplotlib.rcParams['font.serif'] = ['SimHei', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ═══════════════════════════════════════════════════════════════════════════
# ✅ 步骤1：全局配置（跨平台中文字体）
# ═══════════════════════════════════════════════════════════════════════════

def setup_publication_style():
    """
    配置学术论文标准样式

    参考标准：
      - CVPR/ICCV: 使用 Times New Roman 或 Arial
      - NeurIPS: 使用 Computer Modern
      - 推荐 DPI: 300（打印清晰）
    """
    system = platform.system()

    # 字体配置（按优先级）
    if system == 'Windows':
        fonts = ['Times New Roman', 'SimHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        fonts = ['Times New Roman', 'Arial', 'STHeiti']
    else:  # Linux
        fonts = ['Times New Roman', 'DejaVu Serif', 'SimHei']

        # 逐个尝试直到找到
    for font in fonts:
        try:
            matplotlib.rcParams['font.serif'] = [font]
            print(f"✅ 已加载字体: {font}")
            break
        except:
            pass

            # 学术论文标准配置
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.unicode_minus': False,  # 负号正常显示
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'grid.linewidth': 0.8,
        'axes.linewidth': 1.2,  # 坐标轴线宽
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'lines.linewidth': 2.5,  # 数据线宽
        'lines.markersize': 8,
        'pdf.fonttype': 42,  # ← 关键：保证 PDF 字体可编辑
    })

    print("✅ 已应用学术论文标准配置")


setup_publication_style()


# ═══════════════════════════════════════════════════════════════════════════
# ✅ 步骤2：顶会级配色方案
# ═══════════════════════════════════════════════════════════════════════════


# ✅ 辅助函数：计算聚类准确率 (ACC)
# ════════════════════════════════════════════════════════════════════════════════

def calculate_cost_matrix(C, n_clusters):
    """计算Munkres匹配的成本矩阵"""
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    """从Munkres匹配结果提取映射"""
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_accuracy(cluster_assignments, y_true, n_clusters):
    """计算聚类准确率 (ACC)"""
    confusion_mat = confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_mat, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    accuracy = np.mean(y_pred == y_true)
    return accuracy


# ════════════════════════════════════════════════════════════════════════════════
# ✅ 核心函数：肘部法则 - 基于惯性值（Inertia） 二阶差分法
# ════════════════════════════════════════════════════════════════════════════════

def find_optimal_clusters_elbow(
    X: np.ndarray,
    k_range: tuple = (2, 10),
    random_state: int = 42,
    verbose: int = 1
) -> dict:
    """
    使用肘部法则（Elbow Method）寻找最优簇数

    基础指标：惯性值（Inertia）= Σ||x_i - c_k||²
    选择标准：二阶差分最大处（下降速度明显变缓）

    参数：
        X: np.ndarray [N, D]，数据矩阵
        k_range: tuple (min_k, max_k)，搜索的簇数范围
        random_state: int，随机种子
        verbose: int，输出详细度

    返回：dict，包含最优K和所有中间结果
    """
    min_k, max_k = k_range
    inertias = {}
    silhouette_scores = {}
    kmeans_models = {}

    if verbose:
        print(f"\n🔍 使用肘部法则搜索最优簇数 (范围: {min_k}~{max_k})")
        print(f"{'─' * 70}")
        print(f"   K   惯性值      Silhouette   一阶差分   二阶差分   说明")
        print(f"{'─' * 70}")

    # ── Step 1: 训练 K-Means 并收集指标 ──
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        cluster_labels = kmeans.fit_predict(X)
        inertias[k] = kmeans.inertia_
        silhouette_scores[k] = silhouette_score(X, cluster_labels)
        kmeans_models[k] = kmeans

    # ── Step 2: 计算差分（识别肘部）──
    inertia_diffs_1st = {}
    inertia_diffs_2nd = {}

    for k in range(min_k + 1, max_k + 1):
        diff_1st = inertias[k - 1] - inertias[k]
        inertia_diffs_1st[k] = diff_1st

        if k > min_k + 1:
            diff_2nd = inertia_diffs_1st[k - 1] - inertia_diffs_1st[k]
            inertia_diffs_2nd[k] = diff_2nd

    # ── Step 3: 找肘部 ──
    if inertia_diffs_2nd:
        elbow_k = max(inertia_diffs_2nd, key=inertia_diffs_2nd.get)
    else:
        elbow_k = min(inertia_diffs_1st, key=inertia_diffs_1st.get)

    # ── Step 4: 打印详细信息 ──
    if verbose:
        for k in range(min_k, max_k + 1):
            diff_1st = inertia_diffs_1st.get(k, 0)
            diff_2nd = inertia_diffs_2nd.get(k, 0)
            marker = " ← 肘部" if k == elbow_k else ""

        print(f"   {k:2d}  {inertias[k]:10.2f}  {silhouette_scores[k]:8.4f}   "
                  f"{diff_1st:9.2f}   {diff_2nd:9.2f}{marker}")

        print(f"{'─' * 70}")
        print(f"\n✅ 肘部位置: K = {elbow_k}\n")

    return {
        'optimal_k': elbow_k,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'inertia_diffs_1st': inertia_diffs_1st,
        'inertia_diffs_2nd': inertia_diffs_2nd,
        'kmeans_models': kmeans_models,
    }


# ════════════════════════════════════════════════════════════════════════════════
# ✅ 绘图函数：完整的手肘图 + 轮廓系数对比
# ════════════════════════════════════════════════════════════════════════════════
def setup_publication_style():
    """
    配置学术论文标准样式

    参考标准：
      - CVPR/ICCV: 使用 Times New Roman 或 Arial
      - NeurIPS: 使用 Computer Modern
      - 推荐 DPI: 300（打印清晰）
    """
    system = platform.system()

    # 字体配置（按优先级）
    if system == 'Windows':
        fonts = ['Times New Roman', 'SimHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        fonts = ['Times New Roman', 'Arial', 'STHeiti']
    else:  # Linux
        fonts = ['Times New Roman', 'DejaVu Serif', 'SimHei']

        # 逐个尝试直到找到
    for font in fonts:
        try:
            matplotlib.rcParams['font.serif'] = [font]
            print(f"✅ 已加载字体: {font}")
            break
        except:
            pass

            # 学术论文标准配置
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.unicode_minus': False,  # 负号正常显示
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'grid.linewidth': 0.8,
        'axes.linewidth': 1.2,  # 坐标轴线宽
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'lines.linewidth': 2.5,  # 数据线宽
        'lines.markersize': 8,
        'pdf.fonttype': 42,  # ← 关键：保证 PDF 字体可编辑
    })

    print("✅ 已应用学术论文标准配置")


setup_publication_style()


# ═══════════════════════════════════════════════════════════════════════════
# ✅ 步骤2：顶会级配色方案
# ═══════════════════════════════════════════════════════════════════════════

class AcademicColorPalette:
    """
    学术顶会配色方案（参考 CVPR/ICCV/NeurIPS）

    设计原则：
      1. 高对比度：易于区分曲线
      2. 色盲友好：避免红绿组合
      3. 灰度友好：黑白打印仍可识别
    """

    # 主色调（适合论文打印）
    ELBOW_PRIMARY = '#1f77b4'      # 深蓝（肘部法则主线）
    ELBOW_SECONDARY = '#0d47a1'    # 更深蓝（选中点）

    SILHOUETTE_PRIMARY = '#ff7f0e'  # 橙色（轮廓系数）
    SILHOUETTE_SECONDARY = '#d84315' # 深橙

    TRUE_LABEL = '#2ca02c'          # 绿色（真实标签）
    TRUE_SECONDARY = '#1b5e20'      # 深绿

    ELBOW_HIGHLIGHT = '#e53935'     # 红色（肘部标记）

    # 辅助色
    GRID = '#cccccc'
    TEXT = '#333333'  # ✅ 深灰色用于文本

    @staticmethod
    def get_palette():
        """返回整个色板（6个键）"""
        return {
            'elbow': AcademicColorPalette.ELBOW_PRIMARY,
            'elbow_highlight': AcademicColorPalette.ELBOW_HIGHLIGHT,
            'silhouette': AcademicColorPalette.SILHOUETTE_PRIMARY,
            'true_label': AcademicColorPalette.TRUE_LABEL,
            'grid': AcademicColorPalette.GRID,
            'text': AcademicColorPalette.TEXT,  # ✅ 已添加
        }

# ✅ 步骤3：发布级手肘图绘制函数
# ═══════════════════════════════════════════════════════════════════════════

def plot_elbow_publication_quality(
        k_values: np.ndarray,
        inertias: dict,
        silhouette_scores: dict,
        optimal_k: int,
        true_k: int = None,
        output_file: str = 'elbow_analysis.png',
        output_pdf: str = None,
        figsize: tuple = (14, 5),
        dpi: int = 300,
):
    """
    生成学术论文级手肘图

    特性：
      ✅ 符合 CVPR/ICCV/NeurIPS 论文标准
      ✅ 300 DPI，适合印刷
      ✅ PDF + PNG 双格式输出
      ✅ 国际化字体支持
      ✅ 高分辨率数学文本

    参数：
        k_values: K 值数组 [2, 3, 4, ..., 10]
        inertias: 惯性值字典 {k: inertia_value}
        silhouette_scores: 轮廓系数字典 {k: silhouette_value}
        optimal_k: 肘部法选择的最优 K
        true_k: 真实簇数（可选，用于对比）
        output_file: PNG 输出路径
        output_pdf: PDF 输出路径（若为 None 则不输出 PDF）
        figsize: 图表大小
        dpi: 分辨率
    """

    # ────────────────────────────────────────────────────────────
    # 颜色方案
    # ────────────────────────────────────────────────────────────
    palette = AcademicColorPalette.get_palette()

    # ────────────────────────────────────────────────────────────
    # 创建图表
    # ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')

    # ════════════════════════════════════════════════════════════
    # ✅ 左子图：惯性值曲线（肘部法则）
    # ════════════════════════════════════════════════════════════
    ax1 = axes[0]
    inertia_vals = np.array([inertias[k] for k in k_values])

    # ── 主曲线 ──
    line1 = ax1.plot(k_values, inertia_vals, 'o-',
                     color=palette['elbow'],
                     linewidth=2.8,
                     markersize=9,
                     label='Inertia',
                     markeredgecolor=palette['elbow'],
                     markeredgewidth=1.5,
                     zorder=3)

    # ── 填充区域 ──
    ax1.fill_between(k_values, inertia_vals, alpha=0.15,
                     color=palette['elbow'], zorder=1)

    # ── 肘部位置（竖虚线）──
    ax1.axvline(x=optimal_k,
                color=palette['elbow_highlight'],
                linestyle='--',
                linewidth=2.2,
                alpha=0.8,
                zorder=2,
                label=f'Elbow (K = {optimal_k})')

    # ── 肘部点（星形标记）──
    ax1.scatter([optimal_k], [inertias[optimal_k]],
                color=palette['elbow_highlight'],
                s=400,
                marker='*',
                zorder=5,
                edgecolor='#8b0000',
                linewidth=2,
                label=f'Selected elbow')

    # ── 真实簇数（若有）──
    if true_k is not None and true_k in inertias:
        ax1.axvline(x=true_k,
                    color=palette['true_label'],
                    linestyle=':',
                    linewidth=2.2,
                    alpha=0.7,
                    zorder=2)
        ax1.scatter([true_k], [inertias[true_k]],
                    color=palette['true_label'],
                    s=300,
                    marker='s',
                    zorder=5,
                    edgecolor='#1b5e20',
                    linewidth=1.8)
        # ← 手动添加到图例

    # ── 美化 ──
    ax1.set_xlabel('Number of Clusters (K)',
                   fontsize=12, fontweight='bold', color=palette['text'])
    ax1.set_ylabel('Inertia',
                   fontsize=12, fontweight='bold', color=palette['text'])
    ax1.set_title('Elbow Method for Optimal K Selection',
                  fontsize=13, fontweight='bold', pad=15, color=palette['text'])

    ax1.set_xticks(k_values)
    ax1.tick_params(axis='both', which='major', labelsize=11, colors=palette['text'])
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)

    # ── 图例（专业配置）──
    legend1 = ax1.legend(loc='upper right',
                         fontsize=10,
                         framealpha=0.95,
                         edgecolor=palette['text'],
                         fancybox=True,
                         shadow=False,
                         frameon=True)
    legend1.get_frame().set_linewidth(1.2)

    # 若有真实标签，补充图例说明
    if true_k is not None and true_k in inertias:
        ax1.text(0.98, 0.05, f'Ground Truth: K = {true_k}',
                 transform=ax1.transAxes,
                 fontsize=10,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           edgecolor=palette['true_label'],
                           linewidth=1.5,
                           alpha=0.9))

        # ════════════════════════════════════════════════════════════
    # ✅ 右子图：轮廓系数曲线
    # ════════════════════════════════════════════════════════════
    ax2 = axes[1]
    silhouette_vals = np.array([silhouette_scores[k] for k in k_values])
    silhouette_k = max(silhouette_scores, key=silhouette_scores.get)

    # ── 主曲线 ──
    line2 = ax2.plot(k_values, silhouette_vals, 's-',
                     color=palette['silhouette'],
                     linewidth=2.8,
                     markersize=9,
                     label='Silhouette Score',
                     markeredgecolor=palette['silhouette'],
                     markeredgewidth=1.5,
                     zorder=3)

    # ── 填充区域 ──
    ax2.fill_between(k_values, silhouette_vals, alpha=0.15,
                     color=palette['silhouette'], zorder=1)

    # ── 轮廓系数最大值 ──
    ax2.axvline(x=silhouette_k,
                color='#9c27b0',
                linestyle='--',
                linewidth=2.2,
                alpha=0.8,
                zorder=2,
                label=f'Max Silhouette (K = {silhouette_k})')

    ax2.scatter([silhouette_k], [silhouette_scores[silhouette_k]],
                color='#9c27b0',
                s=400,
                marker='*',
                zorder=5,
                edgecolor='#6a1b9a',
                linewidth=2)

    # ── 真实簇数 ──
    if true_k is not None and true_k in silhouette_scores:
        ax2.axvline(x=true_k,
                    color=palette['true_label'],
                    linestyle=':',
                    linewidth=2.2,
                    alpha=0.7,
                    zorder=2)
        ax2.scatter([true_k], [silhouette_scores[true_k]],
                    color=palette['true_label'],
                    s=300,
                    marker='s',
                    zorder=5,
                    edgecolor='#1b5e20',
                    linewidth=1.8)

        # ── 美化 ──
    ax2.set_xlabel('Number of Clusters (K)',
                   fontsize=12, fontweight='bold', color=palette['text'])
    ax2.set_ylabel('Silhouette Coefficient',
                   fontsize=12, fontweight='bold', color=palette['text'])
    ax2.set_title('Silhouette Analysis for Cluster Quality',
                  fontsize=13, fontweight='bold', pad=15, color=palette['text'])

    ax2.set_xticks(k_values)
    ax2.set_ylim(-0.15, 1.0)
    ax2.tick_params(axis='both', which='major', labelsize=11, colors=palette['text'])
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)

    # ── 图例 ──
    legend2 = ax2.legend(loc='upper right',
                         fontsize=10,
                         framealpha=0.95,
                         edgecolor=palette['text'],
                         fancybox=True,
                         shadow=False,
                         frameon=True)
    legend2.get_frame().set_linewidth(1.2)

    if true_k is not None and true_k in silhouette_scores:
        ax2.text(0.98, 0.05, f'Ground Truth: K = {true_k}',
                 transform=ax2.transAxes,
                 fontsize=10,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           edgecolor=palette['true_label'],
                           linewidth=1.5,
                           alpha=0.9))

        # ────────────────────────────────────────────────────────────
    # 全局调整
    # ────────────────────────────────────────────────────────────
    plt.tight_layout(pad=2.5)

    # ────────────────────────────────────────────────────────────
    # 保存输出
    # ────────────────────────────────────────────────────────────

    # PNG 输出（高分辨率）
    plt.savefig(output_file,
                format='png',
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    print(f"✅ PNG 已保存: {output_file} ({dpi} DPI)")

    # PDF 输出（矢量格式，最佳用于论文）
    if output_pdf:
        plt.savefig(output_pdf,
                    format='pdf',
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print(f"✅ PDF 已保存: {output_pdf} (矢量格式，推荐用于论文)")

    plt.close()

    print("\n📊 手肘图绘制完成！")
    print(f"   - PNG 分辨率: {dpi} DPI（适合屏幕显示和在线发表）")
    print(f"   - PDF 格式: 矢量（最佳用于 PDF 论文和编辑修改）")


# ════════════════════════════════════════════════════════════════════════════════
# ✅ 主函数：无监督 K-Means 聚类与评估（多种子版本）
# ════════════════════════════════════════════════════════════════════════════════

def unsupervised_kmeans_elbow_multiseed(
    data_file: str = 'data/correlate_data.npz',
    k_range: tuple = (2, 10),
    n_seeds: int = 10,
    output_file: str = 'unsupervised_kmeans_elbow_results_multiseed.txt',
    plot_file: str = 'elbow_analysis.png',
):
    """
    无监督 K-Means 聚类：多种子肘部法则版本

    主要步骤：
    1. 加载数据（仅特征，不用标签选K）
    2. 对多个随机种子运行肘部法则
    3. 统计K值分布，选择最常见的K
    4. 计算多种子的平均指标
    5. 绘制手肘图
    6. 保存详细结果
    """

    print("=" * 70)
    print("🚀 无监督 K-Means 聚类脚本（多种子肘部法则版本）")
    print("=" * 70)

    # ────────────────────────────────────────────────────────────────
    # ✅ Step 1: 加载数据
    # ────────────────────────────────────────────────────────────────
    print(f"\n📂 Step 1: 加载数据文件 {data_file}")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"❌ 数据文件不存在: {data_file}")

    with np.load(data_file) as data:
        if 'X' in data:
            X = data['X']
            print(f"   ✅ 加载特征数据 'X'，形状: {X.shape}")
        else:
            raise KeyError(f"数据文件中未找到 'X' 键")

        if 'Y' in data:
            Y = data['Y']
            Y = Y - Y.min()
            print(f"   ⚠️  加载标签数据 'Y'（仅用于事后评估）")
        else:
            Y = None
            print(f"   ⚠️  数据中无标签，将采用内在指标评估")

    print(f"\n   数据统计:")
    print(f"   - 样本数 (N)       : {X.shape[0]}")
    print(f"   - 特征数 (D)       : {X.shape[1]}")
    print(f"   - 搜索范围 (K)     : {k_range[0]}~{k_range[1]}")
    print(f"   - 运行种子数       : {n_seeds}")
    if Y is not None:
        true_k = len(np.unique(Y))
        unique_labels = np.unique(Y)
        # ✅ 新增：详细诊断信息
        print(f"   - 真实簇数（Ground Truth）: {true_k}")
        print(f"   - 真实标签值: {unique_labels.tolist()}")

    # ────────────────────────────────────────────────────────────────
    # ✅ Step 2: 多种子运行肘部法则
    # ────────────────────────────────────────────────────────────────
    print(f"\n🔄 Step 2: 运行 {n_seeds} 个随机种子...\n")

    all_seeds_results = []
    k_selections = []

    for seed_idx in range(n_seeds):
        print(f"   [Seed {seed_idx+1:2d}/{n_seeds}]", end=" ")

        # 使用不同的种子运行肘部法则
        opt_result_tmp = find_optimal_clusters_elbow(
            X, k_range=k_range, random_state=seed_idx, verbose=0
        )

        optimal_k_tmp = opt_result_tmp['optimal_k']
        kmeans_model_tmp = opt_result_tmp['kmeans_models'][optimal_k_tmp]
        cluster_labels_tmp = kmeans_model_tmp.fit_predict(X)

        k_selections.append(optimal_k_tmp)

        # 计算该种子的指标
        if Y is not None:
            acc_tmp = get_accuracy(cluster_labels_tmp, Y, optimal_k_tmp)
            ari_tmp = adjusted_rand_score(Y, cluster_labels_tmp)
            nmi_tmp = normalized_mutual_info_score(Y, cluster_labels_tmp)
        else:
            acc_tmp = ari_tmp = nmi_tmp = None

        silhouette_tmp = silhouette_score(X, cluster_labels_tmp)
        dbi_tmp = davies_bouldin_score(X, cluster_labels_tmp)

        result_tmp = {
            'seed': seed_idx,
            'optimal_k': optimal_k_tmp,
            'kmeans_model': kmeans_model_tmp,
            'cluster_labels': cluster_labels_tmp,
            'acc': acc_tmp,
            'ari': ari_tmp,
            'nmi': nmi_tmp,
            'silhouette': silhouette_tmp,
            'dbi': dbi_tmp,
        }
        all_seeds_results.append(result_tmp)
        acc_display = acc_tmp if acc_tmp is not None else 0
        print(f"K={optimal_k_tmp}, ACC={acc_display:.4f}")

        # ────────────────────────────────────────────────────────────────
    # ✅ Step 3: 统计分析
    # ────────────────────────────────────────────────────────────────
    print(f"\n📊 Step 3: 统计结果（{n_seeds}个种子）:\n")

    from collections import Counter
    k_counts = Counter(k_selections)

    print("   肘部选择的K值分布:")
    for k in sorted(k_counts.keys()):
        count = k_counts[k]
        pct = count / n_seeds * 100
        print(f"      K={k}: {count:2d}次 ({pct:5.1f}%)")

    # 最常见的K
    most_common_k = k_counts.most_common(1)[0][0]

    # 选择最常见K对应的结果
    final_result_idx = k_selections.index(most_common_k)

    optimal_k = most_common_k
    kmeans_model = all_seeds_results[final_result_idx]['kmeans_model']
    cluster_labels = all_seeds_results[final_result_idx]['cluster_labels']

    # 计算多种子的平均指标
    if Y is not None:
        accs = np.array([r['acc'] for r in all_seeds_results if r['acc'] is not None])
        aris = np.array([r['ari'] for r in all_seeds_results if r['ari'] is not None])
        nmis = np.array([r['nmi'] for r in all_seeds_results if r['nmi'] is not None])

        print(f"\n   外在指标（均值 ± 标准差）:")
        print(f"      ACC: {np.mean(accs):.6f} ± {np.std(accs):.6f}  "
              f"[{np.min(accs):.4f}, {np.max(accs):.4f}]")
        print(f"      ARI: {np.mean(aris):.6f} ± {np.std(aris):.6f}  "
              f"[{np.min(aris):.4f}, {np.max(aris):.4f}]")
        print(f"      NMI: {np.mean(nmis):.6f} ± {np.std(nmis):.6f}  "
              f"[{np.min(nmis):.4f}, {np.max(nmis):.4f}]")

        acc = np.mean(accs)
        ari = np.mean(aris)
        nmi = np.mean(nmis)
    else:
        acc = ari = nmi = None

    silhouettes = np.array([r['silhouette'] for r in all_seeds_results])
    dbis = np.array([r['dbi'] for r in all_seeds_results])

    print(f"\n   内在指标（均值 ± 标准差）:")
    print(f"      Silhouette: {np.mean(silhouettes):.6f} ± {np.std(silhouettes):.6f}  "
          f"[{np.min(silhouettes):.4f}, {np.max(silhouettes):.4f}]")
    print(f"      DBI:        {np.mean(dbis):.6f} ± {np.std(dbis):.6f}  "
          f"[{np.min(dbis):.4f}, {np.max(dbis):.4f}]")

    silhouette = np.mean(silhouettes)
    dbi = np.mean(dbis)

    # ────────────────────────────────────────────────────────────────
    # ✅ Step 4: 聚类分布
    # ────────────────────────────────────────────────────────────────
    print(f"\n   聚类分布（使用最常见K={optimal_k}）:")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   - Cluster {label}: {count:4d} 样本 ({count/len(cluster_labels)*100:5.1f}%)")

    # ────────────────────────────────────────────────────────────────
    # ✅ Step 5: 绘制手肘图
    # ────────────────────────────────────────────────────────────────
    print(f"\n📈 Step 4: 绘制手肘图")

    # 为绘图重新运行一次肘部法则（使用最常见的种子）
    opt_result = find_optimal_clusters_elbow(
        X, k_range=k_range, random_state=final_result_idx, verbose=0
    )

    k_values = np.array(sorted(opt_result['inertias'].keys()))

    plot_elbow_publication_quality(
        k_values=k_values,
        inertias=opt_result['inertias'],
        silhouette_scores=opt_result['silhouette_scores'],
        optimal_k=optimal_k,
        true_k=len(np.unique(Y)) if Y is not None else None,
        output_file=plot_file.replace('.png', '_publication.png'),
        output_pdf=plot_file.replace('.png', '_publication.pdf'),
        figsize=(14, 5),
        dpi=300,
    )

    # ────────────────────────────────────────────────────────────────
    # ✅ Step 6: 保存结果
    # ────────────────────────────────────────────────────────────────
    print(f"💾 Step 5: 保存结果到文件 {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("无监督 K-Means 聚类结果（多种子肘部法则版本）\n")
        f.write("="*70 + "\n\n")

        f.write("运行配置:\n")
        f.write(f"  随机种子数: {n_seeds}\n")
        f.write(f"  搜索范围: {k_range[0]}~{k_range[1]}\n")
        f.write(f"  每种子初始化: n_init=10\n\n")

        f.write("K值分布:\n")
        for k in sorted(k_counts.keys()):
            count = k_counts[k]
            pct = count / n_seeds * 100
            f.write(f"  K={k}: {count:2d}次 ({pct:5.1f}%)\n")
        f.write(f"  最终选择: K={optimal_k}（最常见）\n\n")

        if Y is not None:
            f.write("外在指标（均值 ± 标准差）:\n")
            f.write(f"  ACC: {np.mean(accs):.6f} ± {np.std(accs):.6f}\n")
            f.write(f"  ARI: {np.mean(aris):.6f} ± {np.std(aris):.6f}\n")
            f.write(f"  NMI: {np.mean(nmis):.6f} ± {np.std(nmis):.6f}\n\n")

        f.write("内在指标（均值 ± 标准差）:\n")
        f.write(f"  Silhouette: {np.mean(silhouettes):.6f} ± {np.std(silhouettes):.6f}\n")
        f.write(f"  DBI:        {np.mean(dbis):.6f} ± {np.std(dbis):.6f}\n\n")

        f.write("详细结果（每个种子）:\n")
        f.write(f"{'Seed':>5} {'K':>3} {'ACC':>10} {'ARI':>10} {'NMI':>10} {'Sil':>10} {'DBI':>10}\n")
        f.write("-"*65 + "\n")
        for r in all_seeds_results:
            f.write(f"{r['seed']:5d} {r['optimal_k']:3d} "
                   f"{r['acc'] if r['acc'] else 0:10.6f} "
                   f"{r['ari'] if r['ari'] else 0:10.6f} "
                   f"{r['nmi'] if r['nmi'] else 0:10.6f} "
                   f"{r['silhouette']:10.6f} {r['dbi']:10.6f}\n")

    print(f"   ✅ 结果已保存\n")

    # ────────────────────────────────────────────────────────────────
    # ✅ 返回结果
    # ────────────────────────────────────────────────────────────────
    results = {
        'optimal_k': optimal_k,
        'cluster_labels': cluster_labels,
        'acc': acc if Y is not None else None,
        'ari': ari if Y is not None else None,
        'nmi': nmi if Y is not None else None,
        'silhouette': silhouette,
        'dbi': dbi,
        'kmeans_model': kmeans_model,
        'data': X,
        'labels': Y,
    }

    return results


# ════════════════════════════════════════════════════════════════════════════════
# ✅ 脚本入口
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='无监督 K-Means 聚类评估（多种子肘部法则版本）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str, default='data/correlate_data.npz',
                        help='数据文件路径')
    parser.add_argument('--k_min', type=int, default=2,
                        help='搜索的最小簇数')
    parser.add_argument('--k_max', type=int, default=10,
                        help='搜索的最大簇数')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='运行的随机种子数')
    parser.add_argument('--output', type=str, default='unsupervised_kmeans_elbow_results_multiseed.txt',
                        help='结果输出文件路径')
    parser.add_argument('--plot', type=str, default='elbow_analysis.png',
                        help='手肘图输出文件路径')

    args = parser.parse_args()

    # 执行无监督聚类
    results = unsupervised_kmeans_elbow_multiseed(
        data_file=args.data,
        k_range=(args.k_min, args.k_max),
        n_seeds=args.n_seeds,
        output_file=args.output,
        plot_file=args.plot,
    )

    print("="*70)
    print("✅ 脚本执行完成！")
    print("="*70)
    print(f"\n🎯 最终结果:")
    print(f"   最优簇数      : {results['optimal_k']}")
    if results['acc'] is not None:
        print(f"   ACC           : {results['acc']:.6f}")
        print(f"   ARI           : {results['ari']:.6f}")
        print(f"   NMI           : {results['nmi']:.6f}")
    print(f"   Silhouette    : {results['silhouette']:.6f}")
    print(f"   DBI           : {results['dbi']:.6f}")
    print(f"\n📁 生成文件:")
    print(f"   {args.output}")
    print(f"   {args.plot}")
# python KMEANS.py --data data/SYN-BIO-6.npz   --n_seeds 10 --k_min 2  --k_max 10
