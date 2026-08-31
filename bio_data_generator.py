# ============================================================
# python bio_data_generator.py --config SYN-BIO-2  --output_dir ./data --seed 123 --visualize
# python bio_data_generator.py --config SYN-BIO-6  --output_dir ./data --seed 123 --visualize
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os


# ================================================================
# 工具函数模块
# ================================================================

def make_correlated_block(
        n_samples: int,
        n_features: int,
        centers: np.ndarray,  # ✅ Fix-3: 从标量改为向量 [n_features]
        within_corr: float = 0.8,
        noise_std: float = 0.1,
        rng: np.random.RandomState = None  # ✅ Fix-4: 新增 rng 参数
) -> np.ndarray:
    """
    生成具有块内强相关性的特征组，模拟生物调控模块共表达。

    ── 生物学背景 ──────────────────────────────────────────────
    在基因组学中，同一调控通路的基因往往受共同转录因子控制，
    表现为"共表达模块"（co-expression module）。
    例如：核糖体蛋白家族（RPS/RPL）在所有翻译活跃的细胞中同步上调。
    流式细胞术中，同一谱系的 CD marker 也呈现高度共表达。

    ── 数学模型（共享隐因子模型）──────────────────────────────
    对于块内第 j 个特征（第 j 个基因/蛋白质）：

        X_{b,j} = center_j + α · Z_shared + (1-α) · ε_j

    其中：
        Z_shared ~ N(centers[0], 0.5²)  : 共享隐因子（通路激活强度）
        ε_j      ~ N(0, noise_std²)     : 独立测量噪声
        α        = sqrt(within_corr)    : 共享因子的加权系数

    块内任意两特征 j, k 的理论相关系数：
        Corr(X_j, X_k) = α² · Var(Z_shared) / (Var(X_j) · Var(X_k))^0.5
                       ≈ within_corr（当 noise_std 较小时近似成立）

    参数：
        n_samples   : 样本数
        n_features  : 块内特征数（同一调控模块的基因/蛋白质数）
        centers     : 每个特征的均值中心，shape=[n_features]
                      ✅ Fix-3: 从标量改为向量，保证块内不同特征有独立的均值，
                      避免所有特征退化到同一中心，增强块间区分度
        within_corr : 块内目标相关系数（0~1），越高表示调控关系越强
        noise_std   : 独立噪声标准差，模拟测量误差
        rng         : 随机状态（✅ Fix-4: 统一 rng 传递，保证种子可控）

    返回：
        block : np.ndarray, shape=[n_samples, n_features]
    """
    if rng is None:
        # 兜底：若未传入 rng，使用全局随机状态（不推荐，破坏可复现性）
        rng = np.random.RandomState(42)

    # α = sqrt(ρ)，使得 Cov(X_j, X_k) = α² · Var(Z) ≈ ρ
    alpha = np.sqrt(within_corr)

    # 共享隐因子：以第一个特征的中心为基准生成
    # 生物意义：通路激活强度，决定整个模块的整体表达水平
    Z_shared = rng.normal(centers[0], 0.5, n_samples)  # shape=[n_samples]

    block = np.zeros((n_samples, n_features))
    for j in range(n_features):
        # 每个特征的独立测量噪声（如测序随机误差、荧光背景噪声）
        epsilon_j = rng.normal(0, noise_std, n_samples)

        # ✅ Fix-3: centers[j % len(centers)] 确保每个特征使用自己对应的中心
        # 修复前：block[:, j] = alpha * Z_shared + (1-alpha) * epsilon_j
        #         （所有特征共享同一个标量 center，特征间均值无差异）
        # 修复后：每个特征有独立中心，保留簇中心的空间信息
        block[:, j] = (
                centers[j % len(centers)]  # 该特征在当前簇的均值中心
                + alpha * Z_shared  # 共享通路信号（产生块内相关性）
                + (1 - alpha) * epsilon_j  # 独立噪声（降低块内相关性）
        )

    return block  # shape=[n_samples, n_features]


def add_zero_inflation(
        X: np.ndarray,
        dropout_rate: float = 0.3,
        rng: np.random.RandomState = None  # ✅ Fix-4: 新增 rng 参数
) -> np.ndarray:
    """
    模拟 scRNA-seq 数据的零膨胀（Zero Inflation / Dropout）效应。

    ── 生物学背景 ──────────────────────────────────────────────
    单细胞 RNA 测序（scRNA-seq）中，由于单细胞 RNA 含量极少（~10pg），
    以下技术因素导致大量"假零值"（dropout）：

        1. 细胞裂解效率     : ~60%（40% mRNA 直接丢失）
        2. 逆转录捕获效率   : 10%~40%（低表达基因极易丢失）
        3. PCR 扩增偏差     : 低丰度转录本被高丰度转录本竞争排挤
        4. 测序深度限制     : 每个细胞仅测序 ~2000~5000 个基因（人类共~20000个）

    这些因素叠加，导致 scRNA-seq 数据矩阵中：
        - 零值比例通常达 60%~95%（普通 RNA-seq 约 20%）
        - 零值分两类：
            * 结构性零（真实不表达）: 生物学真实，含判别信息
            * 随机性零（技术 dropout）: 技术噪声，应区分处理

    ── 数学模型（伯努利 Dropout 模型）──────────────────────────
    对于样本 i、特征 j 的观测值：

        X̃_{ij} = X_{ij} · B_{ij}

    其中：
        B_{ij} ~ Bernoulli(1 - p_dropout)
               = { 0  以概率 p_dropout    （技术性丢失）
                   1  以概率 1-p_dropout  （正常观测）  }

    注意：此处简化了 ZINB（零膨胀负二项）完整模型，
    完整模型为：
        P(X=k) = π·I(k=0) + (1-π)·NB(k; μ, φ)
    本函数对应其中的 dropout 部分（π 对应 p_dropout）。

    ── 为何必须在 StandardScaler 之前施加 ──────────────────────
    若先 StandardScaler 再 add_zero_inflation（错误顺序）：
        标准化后 X̃ ∈ N(0,1)，零值被强制赋为 0，
        但标准化后的均值已非0，零值不再代表"缺失"。

    正确顺序（Fix-2）：
        add_batch_effect → add_zero_inflation → [无 StandardScaler]
        零值在原始尺度上置零，保留"基因未被检测到"的语义。
        dataset.py 的 StandardScaler 会将零值映射为 -μ/σ < 0，
        形成可识别的"缺失表达"信号，GatingNet 可学习此模式。

    参数：
        X            : 输入特征矩阵 [N, D]（原始表达量）
        dropout_rate : 技术性零值的比例（伯努利概率 p）
                       SYN-BIO-2（PBMC）: 0.20
                       SYN-BIO-5（TME）:  0.35
        rng          : 随机状态（✅ Fix-4: 统一传递，保证可复现）

    返回：
        X_zi : np.ndarray [N, D]，含零膨胀的表达矩阵
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # 生成伯努利 mask：True 表示该值被技术性 dropout（置零）
    # rng.rand(*X.shape) 生成 [0,1) 均匀分布，< dropout_rate 的位置置零
    mask = rng.rand(*X.shape) < dropout_rate  # shape=[N, D], dtype=bool

    X_zi = X.copy()
    X_zi[mask] = 0.0  # 技术性 dropout：将被选中的位置强制置零

    return X_zi  # shape=[N, D]


def add_batch_effect(
        X: np.ndarray,
        # ✅ Fix-6: 移除无用的 Y 参数（原函数接收 Y 但从未使用）
        # 原签名: def add_batch_effect(X, Y, n_batches, batch_std)
        n_batches: int = 2,
        batch_std: float = 0.3,
        rng: np.random.RandomState = None  # ✅ Fix-4: 新增 rng 参数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟批次效应（Batch Effect）。

    ── 生物学背景 ──────────────────────────────────────────────
    批次效应是高通量生物学实验中普遍存在的系统性偏差，来源包括：

        1. 不同测序批次  : 试剂批号差异、测序仪状态不同
        2. 样本处理时间  : 不同天采集的样本细胞状态略有差异
        3. 操作人员差异  : 不同实验员的手法引入操作偏差
        4. 测序深度差异  : 不同批次的 reads 数量不同

    批次效应的统计特性：
        - 同一批次内的所有样本受到相同的系统偏移
        - 批次偏移通常被建模为加性高斯噪声：
            X_batch = X_true + b_batch
            其中 b_batch ~ N(0, σ_batch²·I)

    ── 为何在 add_zero_inflation 之前施加 ───────────────────────
    批次效应是在数据产生时（测序过程中）引入的，
    而 dropout（零膨胀）发生在更早的样本处理阶段。

    正确的数据生成顺序应模拟真实实验流程：
        真实表达量 → 批次效应（测序偏差）→ dropout（捕获失败）

    ── 数学模型 ────────────────────────────────────────────────
    设 n_batches=2（批次0和批次1），批次偏移向量：
        b_0 ~ N(0, σ²·I_D)
        b_1 ~ N(0, σ²·I_D)

    样本 i 属于批次 t_i（均匀随机分配），则：
        X̃_{i,j} = X_{i,j} + b_{t_i, j}

    参数：
        X         : 输入特征矩阵 [N, D]
        n_batches : 批次数量（默认2，模拟两次测序批次）
        batch_std : 批次偏移的标准差（控制批次效应强度）
                    0.2 表示偏移约为特征标准差的 20%
        rng       : 随机状态（✅ Fix-4: 统一传递）

    返回：
        X_batch      : np.ndarray [N, D]，添加批次效应后的矩阵
        batch_labels : np.ndarray [N]，每个样本所属的批次编号
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_samples = X.shape[0]

    # 随机将样本分配到不同批次（均匀随机分配，模拟随机采样设计）
    batch_labels = rng.randint(0, n_batches, n_samples)  # shape=[N]

    # 为每个批次生成独立的系统性偏移向量
    # 每个特征在不同批次有不同的偏移量（模拟试剂批号对不同基因的影响）
    batch_shifts = rng.normal(0, batch_std, (n_batches, X.shape[1]))  # shape=[n_batches, D]

    X_batch = X.copy()
    for b in range(n_batches):
        idx = batch_labels == b  # 属于批次 b 的样本索引
        X_batch[idx] += batch_shifts[b]  # 对该批次所有样本施加相同的系统偏移

    return X_batch, batch_labels  # ([N, D], [N])


# ================================================================
# 主数据生成器
# ================================================================

class BioDataGenerator:
    """
    生物医学背景模拟数据集生成器。

    设计目标：
        为 IDC+SEFS 框架的验证实验提供具有真实生物学统计特性的
        合成数据集，覆盖从低维（D=15）到超高维（D=2000）、
        从大样本（N=10000）到小样本（N=200）的多种场景。

    特征生成结构（每个簇内）：
        ┌─────────────────────────────────────────────────────┐
        │  Part A: 相关特征块（n_corr_blocks × block_size）   │
        │          → 模拟基因调控模块/蛋白质复合体            │
        │          → within_corr 控制块内相关强度             │
        ├─────────────────────────────────────────────────────┤
        │  Part B: 独立信息性特征（n_informative - Part A）   │
        │          → 模拟单个判别性 marker（如 CD4、CD8）     │
        ├─────────────────────────────────────────────────────┤
        │  Part C: 纯噪声特征（n_features - n_informative）   │
        │          → 模拟背景基因/冗余特征                    │
        │          → SEFS 应学会忽略这些特征                  │
        └─────────────────────────────────────────────────────┘

    接口：
        - 保存格式：np.savez(fname, X=X, Y=Y)
        - NumpyTableDataset.setup(filepath) 可直接读取 'X'/'Y' 键
        - 不在此处做 StandardScaler（由 dataset.py 统一处理）
    """

    # ── 七种生物学数据集配置 ──────────────────────────────────────
    CONFIGS = {
        # ────────────────────────────────────────────────────────
        # SYN-BIO-1: 流式细胞术 Surface Marker Panel
        # 生物背景：免疫细胞分型（CD3/CD4/CD8/CD19/CD56等），低维强相关
        # D/N=0.04（远低于1），OAS收缩强度低，相关结构清晰
        # ────────────────────────────────────────────────────────
        'SYN-BIO-1': {
            'n_samples': 500,
            'n_features': 20,
            'n_clusters': 4,
            'bio_context': 'Flow Cytometry Surface Markers',
            'noise_std': 0.15,
            'has_zero_inflation': False,  # 流式不存在 dropout
            'has_batch_effect': True,  # 不同批次抗体标记效率不同
            'n_informative': 8,  # 8个 CD marker 有判别力
            'n_corr_blocks': 3,  # 3个谱系模块（T/B/NK）
            'block_size': 3,  # 每模块3个 marker
            'within_corr': 0.85,  # CD marker 块内强相关
            'cluster_sep': 2.0,  # 免疫细胞亚型分离度较高
            'recommended_corr_threshold': 0.6,
        },

        # ────────────────────────────────────────────────────────
        # SYN-BIO-2: 单细胞 RNA-seq PBMC（小规模）
        # 生物背景：外周血单核细胞5亚型，中高维稀疏数据
        # 含20% dropout（低表达基因测序捕获失败）
        # ────────────────────────────────────────────────────────
        'SYN-BIO-2': {
            'n_samples': 800,
            'n_features': 100,
            'n_clusters': 5,
            'bio_context': 'scRNA-seq PBMC Immune Subtypes',
            'noise_std': 0.2,
            'has_zero_inflation': True,
            'dropout_rate': 0.20,  # PBMC 测序质量较好，dropout 较低
            'has_batch_effect': True,  # 不同测序批次
            'n_informative': 30,  # 30个细胞类型标志基因
            'n_corr_blocks': 6,  # 6个基因共调控模块
            'block_size': 5,  # 每模块5个基因
            'within_corr': 0.75,
            'cluster_sep': 1.5,
            'recommended_corr_threshold': 0.5,
        },

        # ────────────────────────────────────────────────────────
        # SYN-BIO-3: 原版基线（对应 correlate_data.py，供对照）
        # 生物背景：小型批量RNA-seq实验（4组处理条件）
        # ────────────────────────────────────────────────────────
        'SYN-BIO-3': {
            'n_samples': 3200,
            'n_features': 15,
            'n_clusters': 4,
            'bio_context': 'Baseline (Original IDC SYN-C1)',
            'noise_std': 0.1,
            'has_zero_inflation': False,  # 批量 RNA-seq dropout 可忽略
            'has_batch_effect': False,  # 原版基线不含批次效应
            'n_informative': 5,  # 对应原版的5个有效特征
            'n_corr_blocks': 2,  # 2个相关对（X1-X4线性，X3-X5非线性）
            'block_size': 2,
            'within_corr': 0.70,
            'cluster_sep': 2.0,
            'recommended_corr_threshold': 0.3,
        },

        # ────────────────────────────────────────────────────────
        # SYN-BIO-4: 蛋白质组学 TMT 定量（CPTAC-like）
        # 生物背景：癌症蛋白质组学，蛋白质复合体共表达模块
        # N=5000 >> D=200，OAS收缩极低，相关结构被完整保留
        # ────────────────────────────────────────────────────────
        'SYN-BIO-4': {
            'n_samples': 5000,
            'n_features': 200,
            'n_clusters': 6,
            'bio_context': 'Proteomics TMT Quantification (CPTAC-like)',
            'noise_std': 0.12,  # TMT 质谱噪声低于 RNA-seq
            'has_zero_inflation': False,  # 蛋白质组学无 dropout 效应
            'has_batch_effect': True,  # TMT 标记批次差异
            'n_informative': 60,  # 60个癌症相关蛋白
            'n_corr_blocks': 10,  # 10个蛋白质复合体模块
            'block_size': 6,  # 每模块6个亚基
            'within_corr': 0.80,  # 蛋白质复合体亚基强共表达
            'cluster_sep': 1.8,
            'recommended_corr_threshold': 0.55,
        },

        # ────────────────────────────────────────────────────────
        # SYN-BIO-5: 单细胞 RNA-seq 肿瘤微环境（中等规模）
        # 生物背景：TME 8种细胞类型，高维稀疏（dropout=35%）
        # 最复杂的生物学场景：细胞类型多、dropout率高、样本量大
        # ────────────────────────────────────────────────────────
        'SYN-BIO-5': {
            'n_samples': 10000,
            'n_features': 500,
            'n_clusters': 8,
            'bio_context': 'scRNA-seq Tumor Microenvironment (TME)',
            'noise_std': 0.25,
            'has_zero_inflation': True,
            'dropout_rate': 0.35,  # 肿瘤组织解离损伤导致高 dropout
            'has_batch_effect': True,  # 不同患者样本的批次效应
            'n_informative': 120,  # 120个细胞类型 marker 基因
            'n_corr_blocks': 16,  # 16个功能基因模块
            'block_size': 8,  # 每模块8个基因（通路更大）
            'within_corr': 0.70,
            'cluster_sep': 1.3,  # 8类细胞分离度较低
            'recommended_corr_threshold': 0.45,
        },

        # ────────────────────────────────────────────────────────
        # SYN-BIO-6: 基因芯片高维小样本（SRBCT-like）
        # 生物背景：4种小儿实体瘤（EWS/BL/NB/RMS），D>>N 极端场景
        # D/N=10，OAS收缩强度 α>0.9，验证框架极端鲁棒性
        # ────────────────────────────────────────────────────────
        'SYN-BIO-6': {
            'n_samples': 200,
            'n_features': 2000,
            'n_clusters': 4,
            'bio_context': 'Gene Expression Microarray (SRBCT-like, D>>N)',
            'noise_std': 0.3,
            'has_zero_inflation': False,  # 微阵列无 dropout
            'has_batch_effect': True,  # 不同芯片批次
            'n_informative': 80,  # 80个肿瘤类型特异性基因
            'n_corr_blocks': 16,  # 16个基因调控模块
            'block_size': 5,
            'within_corr': 0.65,  # D>>N 时相关估计已较弱
            'cluster_sep': 1.2,  # 4种形态相似肿瘤难以区分
            'recommended_corr_threshold': 0.4,
        },

        # ────────────────────────────────────────────────────────
        # SYN-BIO-7: 泛癌多组学（TCGA-like）
        # 生物背景：5种癌症亚型的 mRNA+miRNA+甲基化联合分析
        # 含 miRNA-mRNA 负调控对，相关结构包含正负相关混合
        # ────────────────────────────────────────────────────────
        'SYN-BIO-7': {
            'n_samples': 800,
            'n_features': 1000,
            'n_clusters': 5,
            'bio_context': 'Pan-Cancer Multi-Omics (TCGA-like)',
            'noise_std': 0.2,
            'has_zero_inflation': False,  # 多组学数据已预处理
            'has_batch_effect': True,  # 不同癌种样本采集批次差异
            'n_informative': 200,  # 200个多组学判别特征
            'n_corr_blocks': 20,  # 20个功能模块（含 miRNA-mRNA 对）
            'block_size': 10,  # 每模块10个特征（多组学联合）
            'within_corr': 0.72,
            'cluster_sep': 1.4,
            'recommended_corr_threshold': 0.45,
        },
    }

    def __init__(self, config_name: str = 'SYN-BIO-3', seed: int = 42):
        """
        初始化并生成数据集。

        参数：
            config_name : 数据集配置名（见 CONFIGS 字典）
            seed        : 全局随机种子（保证跨平台可复现）

        执行流程：
            __init__ → _generate() → _print_stats()
        """
        if config_name not in self.CONFIGS:
            raise ValueError(
                f"未知配置 '{config_name}'，"
                f"可选: {list(self.CONFIGS.keys())}"
            )

        self.config_name = config_name
        self.cfg = self.CONFIGS[config_name]
        self.seed = seed
        # ✅ Fix-4: 使用 RandomState 对象统一管理随机状态
        # 避免使用全局 np.random（全局状态会被其他代码污染）
        self.rng = np.random.RandomState(seed)

        self.X, self.Y = self._generate()
        self._print_stats()

    # ────────────────────────────────────────────────────────────
    # 核心数据生成逻辑
    # ────────────────────────────────────────────────────────────
    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据生成主流程。

        生成步骤（修复后的正确顺序）：
            Step 1: 生成 K 个正交化簇中心（QR 分解保证分离度）
            Step 2: 对每个簇生成特征矩阵（含相关块+独立特征+噪声）
            Step 3: 添加批次效应（在原始尺度上，物理意义最清晰）
            Step 4: 添加零膨胀（在标准化之前，保留零值语义）
            [✅ Fix-1: 移除 StandardScaler，由 dataset.py 统一处理]
            Step 5: 打乱样本顺序（消除簇间的序号偏差）

        返回：
            X : np.ndarray [N, D], float32（原始未标准化数据）
            Y : np.ndarray [N], int64（类别标签，0~K-1）
        """
        cfg = self.cfg
        N = cfg['n_samples']
        D = cfg['n_features']
        K = cfg['n_clusters']
        n_per_cluster = N // K  # 每个簇的样本数（整除部分）

        # ── Step 1: 生成 K 个正交化簇中心 ──────────────────────────
        centers = self._make_cluster_centers(K, D, cfg['cluster_sep'])
        # centers.shape = [K, D]

        X_list, Y_list = [], []

        for k in range(K):
            # 最后一个簇处理样本数的余数，保证总样本数精确等于 N
            n_k = n_per_cluster if k < K - 1 else N - n_per_cluster * (K - 1)

            # ── Step 2: 生成第 k 个簇的特征矩阵 ──
            X_k = self._generate_cluster(k, n_k, D, centers[k], cfg)
            # X_k.shape = [n_k, D]

            X_list.append(X_k)
            Y_list.extend([k] * n_k)

        X = np.vstack(X_list).astype(np.float32)  # [N, D]
        Y = np.array(Y_list, dtype=np.int64)  # [N]

        # ── Step 3: 添加批次效应（在原始表达尺度上施加）────────────
        # ✅ Fix-2: 批次效应在零膨胀之前施加
        # 理由：批次偏移发生在测序过程（实验层面），
        #       dropout 发生在样本处理阶段（更早），
        #       但统计建模时，先施加系统性偏移再模拟随机丢失更合理
        if cfg.get('has_batch_effect', False):
            X, _ = add_batch_effect(
                X,
                n_batches=2,
                batch_std=0.2,
                rng=self.rng  # ✅ Fix-4: 传入 rng
            )

        # ── Step 4: 添加零膨胀（在标准化之前施加，保留零值语义）──
        # ✅ Fix-2: 零膨胀在批次效应之后、标准化之前施加
        # 关键原因：若先 StandardScaler 再 add_zero_inflation，
        #   标准化后 X ~ N(0,1)，置零后零值均值偏差 = -μ/σ 的负数，
        #   而非真正意义上的"未检测到"（在原始尺度上为0）
        if cfg.get('has_zero_inflation', False):
            X = add_zero_inflation(
                X,
                cfg.get('dropout_rate', 0.3),
                rng=self.rng  # ✅ Fix-4: 传入 rng
            )

        # ── [✅ Fix-1] 移除此处的 StandardScaler ────────────────────
        # 原代码：X = StandardScaler().fit_transform(X)
        #
        # 移除原因：
        #   dataset.py 的 NumpyTableDataset.setup() 中已有：
        #       X = preprocessing.StandardScaler().fit_transform(X)   [^8]
        #   若此处再做一次，形成双重标准化：
        #       第一次（此处）：X → N(0,1)
        #       第二次（dataset.py）：对已是 N(0,1) 的数据再做标准化
        #   虽然第二次理论上近似恒等变换，但：
        #     1. 零膨胀数据：零值已被第一次标准化扭曲为负数，第二次进一步混乱
        #     2. OAS 协方差估计：基于两次标准化后的数据，相关结构估计有偏
        #     3. 高维小样本（D>>N）：样本均值/方差不稳定，两次误差累积

        # ── Step 5: 打乱样本顺序 ──────────────────────────────────
        # ✅ Fix-4: 使用 self.rng 而非全局 np.random.permutation
        shuffle_idx = self.rng.permutation(N)
        X = X[shuffle_idx]
        Y = Y[shuffle_idx]

        return X, Y

    def _make_cluster_centers(
            self,
            K: int,
            D: int,
            sep: float,
    ) -> np.ndarray:
        """
        在 D 维空间中生成 K 个正交化的簇中心。

        ── 数学原理（QR 分解正交化）───────────────────────────────
        直接随机生成 K 个中心可能导致中心间距离不均匀，
        甚至出现两个中心很近的情况（影响聚类难度控制）。

        使用 QR 分解保证正交性：
            raw ~ N(0, I_{D×K})    # 随机初始矩阵
            Q, R = QR(raw^T)       # QR 分解，Q 的列为正交基向量
            centers = Q[:, :K]^T  # 取前 K 列，形成 K 个正交方向

        正交基向量满足：
            ||center_k|| = 1（单位向量）
            center_i · center_j = 0（i≠j，相互正交）

        乘以 sep · sqrt(D) 控制簇间距离：
            ||center_i - center_j|| = sqrt(2) · sep · sqrt(D)
            → sep 越大，簇间欧式距离越大，分类越容易

        参数：
            K   : 簇数量
            D   : 特征维度
            sep : 簇间分离度系数（越大越容易分类）

        返回：
            centers : np.ndarray [K, D]
        """
        # 生成随机初始矩阵（shape=[D, K]）
        raw = self.rng.randn(K, D)  # [K, D]

        # QR 分解：raw^T = Q @ R，Q 的列为正交单位向量
        Q, _ = np.linalg.qr(raw.T)  # Q.shape=[D, K]（或[D,D] if K<=D）

        # 取前 K 列正交向量作为簇中心，缩放到 sep·sqrt(D) 的距离
        centers = Q[:, :K].T * sep * np.sqrt(D)  # [K, D]

        return centers

    def _generate_cluster(
            self,
            cluster_id: int,
            n_samples: int,
            n_features: int,
            center: np.ndarray,  # shape=[D]，该簇在每个特征上的均值中心
            cfg: dict,
    ) -> np.ndarray:
        """
        生成单个簇的特征矩阵。

        特征分为三类（对应生物学中不同类型的基因/蛋白质）：

        Part A: 相关特征块（基因调控模块）
            - n_corr_blocks 个模块，每个 block_size 个特征
            - 模块内特征高度相关（within_corr）
            - 模拟：核糖体蛋白模块/细胞周期基因模块/细胞因子模块

        Part B: 独立信息性特征（单个 marker 基因）
            - n_informative - n_block_feats 个独立特征
            - 在簇中心处有偏移，具有判别力，但特征间无相关性
            - 模拟：CD4/CD8 等单独的表面标志物

        Part C: 纯噪声特征（背景基因）
            - n_features - n_informative 个噪声特征
            - 均值为0，方差为 (2*noise_std)²，无判别力
            - 模拟：管家基因/随机背景转录本
            - SEFS 的目标之一：抑制这些特征的门控值

        参数：
            cluster_id : 簇编号（0~K-1）
            n_samples  : 该簇的样本数
            n_features : 总特征数 D
            center     : 该簇的中心向量 [D]
            cfg        : 数据集配置字典

        返回：
            X_k : np.ndarray [n_samples, n_features]
        """
        n_blocks = cfg['n_corr_blocks']
        block_sz = cfg['block_size']
        within_corr = cfg['within_corr']
        noise_std = cfg['noise_std']
        n_informative = cfg['n_informative']

        # 计算三类特征的数量
        n_block_feats = n_blocks * block_sz  # Part A 的特征数
        n_indep_feats = max(0, n_informative - n_block_feats)  # Part B
        n_noise_feats = n_features - n_informative  # Part C

        X_k = np.zeros((n_samples, n_features))
        feat_idx = 0  # 当前填充到的特征索引

        # ── Part A: 相关特征块（基因调控模块）─────────────────────
        for b in range(n_blocks):
            if feat_idx + block_sz > n_features:
                break  # 防止超出特征维度（高维数据集不会触发，低维可能触发）

            # ✅ Fix-3: 提取该块对应的特征中心向量（而非标量）
            # 修复前：block_center = center[feat_idx]（标量，所有特征共享）
            # 修复后：block_centers = center[feat_idx:feat_idx+block_sz]（向量）
            block_centers = center[feat_idx: feat_idx + block_sz]  # [block_sz]

            block = make_correlated_block(
                n_samples=n_samples,
                n_features=block_sz,
                centers=block_centers,  # ✅ Fix-3: 向量化中心
                within_corr=within_corr,
                noise_std=noise_std,
                rng=self.rng  # ✅ Fix-4: 传入 rng
            )
            X_k[:, feat_idx: feat_idx + block_sz] = block
            feat_idx += block_sz

        # ── Part B: 独立信息性特征（单个判别性 marker）────────────
        for _ in range(n_indep_feats):
            if feat_idx >= n_features:
                break
            # 在对应簇中心处加独立高斯噪声
            X_k[:, feat_idx] = (
                    center[feat_idx]
                    + self.rng.normal(0, noise_std, n_samples)  # ✅ Fix-4
            )
            feat_idx += 1

        # ── Part C: 纯噪声特征（背景基因，无判别力）───────────────
        # 使用 2*noise_std 的标准差（比信息性特征噪声更大）
        # 这些特征在所有簇中均值为0，SEFS 应学会将其门控值压制至0
        for _ in range(n_noise_feats):
            if feat_idx >= n_features:
                break
            X_k[:, feat_idx] = self.rng.normal(0, noise_std * 2, n_samples)  # ✅ Fix-4
            feat_idx += 1

        return X_k  # [n_samples, n_features]

    # ────────────────────────────────────────────────────────────
    # 统计信息打印
    # ────────────────────────────────────────────────────────────
    def _print_stats(self):
        """打印数据集的关键统计信息，包含修复状态提示。"""
        cfg = self.cfg
        print(f"\n{'=' * 65}")
        print(f"  [{self.config_name}] {cfg['bio_context']}")
        print(f"{'=' * 65}")
        print(f"  样本数 N                    : {self.X.shape[0]}")
        print(f"  特征数 D                    : {self.X.shape[1]}")
        print(f"  类别数 K                    : {cfg['n_clusters']}")
        print(f"  D/N 比                      : {self.X.shape[1] / self.X.shape[0]:.3f}")
        print(f"  信息性特征数                : {cfg['n_informative']} / {self.X.shape[1]}")
        print(f"  相关特征块                  : "
              f"{cfg['n_corr_blocks']} 块 × {cfg['block_size']} 特征/块")
        print(f"  块内相关强度                : {cfg['within_corr']}")
        print(f"  零膨胀(dropout)             : "
              f"{cfg.get('has_zero_inflation', False)}", end='')
        if cfg.get('has_zero_inflation'):
            print(f" (rate={cfg.get('dropout_rate', 0.3):.0%})")
        else:
            print()
        print(f"  批次效应                    : {cfg.get('has_batch_effect', False)}")

        # ✅ Fix-1 状态说明
        print(f"  ⚠️  StandardScaler          : 未在此处执行")
        print(f"     → 由 dataset.py 的 NumpyTableDataset.setup() 统一处理")
        print(f"     → 避免与 dataset.py 的双重标准化问题")

        print(f"  数据值域（未标准化）        : "
              f"[{self.X.min():.3f}, {self.X.max():.3f}]")
        print(f"  推荐 correlation_threshold  : "
              f"{cfg.get('recommended_corr_threshold', 0.3)}")
        print()

        # 类别分布（检查是否均衡）
        unique, counts = np.unique(self.Y, return_counts=True)
        print("  类别分布：")
        for cls, cnt in zip(unique, counts):
            print(f"    类别 {cls}: {cnt:5d} 样本 ({cnt / len(self.Y) * 100:.1f}%)")

    # ────────────────────────────────────────────────────────────
    # 可视化（PCA / t-SNE 降维散点图）
    # ────────────────────────────────────────────────────────────
    def visualize(self, method: str = 'pca'):
        """
        可视化数据集的簇结构（降维到2D）。

        参数：
            method : 'pca'  → 适合低维/线性可分数据（D < 100）
                     'tsne' → 适合高维/非线性数据（D ≥ 100，但速度慢）

        注意：
            - 高维数据（D=2000）t-SNE 可能需要几分钟
            - SYN-BIO-4/5/7 建议先用 PCA 降到 50 维再做 t-SNE
        """
        from sklearn.decomposition import PCA
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            method = 'pca'

        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor('w')

        if method == 'pca' or self.X.shape[1] <= 2:
            if self.X.shape[1] > 2:
                reducer = PCA(n_components=2, random_state=self.seed)
                X_2d = reducer.fit_transform(self.X)
                var_ratio = reducer.explained_variance_ratio_.sum()
                method_label = f'PCA (explained var={var_ratio:.1%})'
            else:
                X_2d = self.X
                method_label = 'Raw (D=2)'
        else:
            # 高维数据先 PCA 到 50 维，加速 t-SNE
            if self.X.shape[1] > 50:
                pca_50 = PCA(n_components=50, random_state=self.seed)
                X_pca = pca_50.fit_transform(self.X)
            else:
                X_pca = self.X
            reducer = TSNE(n_components=2, random_state=self.seed, perplexity=30)
            X_2d = reducer.fit_transform(X_pca)
            method_label = 't-SNE'

        K = self.cfg['n_clusters']
        cmap = plt.cm.get_cmap('tab10', K)
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=self.Y, cmap=cmap, s=15,
            alpha=0.6, edgecolors='none'
        )
        ax.set_title(
            f"[{self.config_name}] {self.cfg['bio_context']}\n"
            f"N={self.X.shape[0]}, D={self.X.shape[1]}, "
            f"K={K} ({method_label})",
            fontsize=12
        )
        ax.set_xlabel('Dim 1', fontsize=12)
        ax.set_ylabel('Dim 2', fontsize=12)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(i), markersize=8)
            for i in range(K)
        ]
        ax.legend(handles, [f'Cluster {i}' for i in range(K)],
                  loc='best', fontsize=9)
        plt.tight_layout()

        fname = f"{self.config_name}_visualization.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化已保存: {fname}")
        plt.show()

    # ────────────────────────────────────────────────────────────
    # 保存为 .npz（与 NumpyTableDataset.setup 接口完全兼容）
    # ────────────────────────────────────────────────────────────
    def save(self, output_dir: str = 'data') -> str:
        """
        保存数据集为 .npz 格式，并生成配套的 yaml 配置提示文件。

        保存格式（与 correlate_data.py 完全一致）：
            np.savez(fname, X=self.X, Y=self.Y)

        NumpyTableDataset.setup() 中优先读取 'X' 键，可直接加载[^8]：
            data_dict = np.load(filepath)
            X = data_dict['X']  # ← 匹配此处的保存格式
            Y = data_dict['Y']

        重要说明：
            保存的 X 是未经 StandardScaler 的原始数据（Fix-1）。
            dataset.py 的 setup() 会在加载后执行 StandardScaler。

        参数：
            output_dir : 输出目录（不存在则自动创建）

        返回：
            fname : 保存的 .npz 文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"{self.config_name}.npz")

        # 保存原始未标准化数据（dataset.py 会做标准化）
        np.savez(fname, X=self.X, Y=self.Y)

        # ── 生成配套 yaml 配置提示文件 ──────────────────────────
        # 方便用户直接复制 recommended_corr_threshold 到 cfg.yaml
        yaml_hint_path = os.path.join(output_dir, f"{self.config_name}_hint.yaml")
        cfg = self.cfg
        with open(yaml_hint_path, 'w', encoding='utf-8') as f:
            f.write(f"# 自动生成 - {self.config_name} 配置提示\n")
            f.write(f"# 生物背景: {cfg['bio_context']}\n")
            f.write(f"# N={cfg['n_samples']}, D={cfg['n_features']}, "
                    f"K={cfg['n_clusters']}, "
                    f"D/N={cfg['n_features'] / cfg['n_samples']:.3f}\n\n")
            f.write(f"data_file: {fname}\n")
            f.write(f"correlation_threshold: "
                    f"{cfg.get('recommended_corr_threshold', 0.3)}\n")

        print(f"✅ 数据已保存       : {fname}  "
              f"(X={self.X.shape}, Y={self.Y.shape})")
        print(f"✅ 配置提示已保存   : {yaml_hint_path}")
        return fname


# ================================================================
# 批量生成所有数据集
# ================================================================

def generate_all_datasets(output_dir: str = 'data', seed: int = 42) -> list:
    """
    一键生成全部 7 种数据集并保存。

    参数：
        output_dir : 输出目录
        seed       : 随机种子

    返回：
        summary : 包含每个数据集元信息的字典列表
    """
    print("\n🚀 开始批量生成生物医学模拟数据集...")
    summary = []

    for name in BioDataGenerator.CONFIGS.keys():
        gen = BioDataGenerator(config_name=name, seed=seed)
        fpath = gen.save(output_dir=output_dir)
        summary.append({
            'name': name,
            'N': gen.X.shape[0],
            'D': gen.X.shape[1],
            'K': gen.cfg['n_clusters'],
            'D/N': f"{gen.X.shape[1] / gen.X.shape[0]:.3f}",
            'context': gen.cfg['bio_context'],
            'fpath': fpath,
        })

    # 打印汇总表
    print(f"\n{'=' * 80}")
    print("📋 数据集汇总（✅ Fix-1: 所有数据集均未做 StandardScaler）")
    print(f"{'=' * 80}")
    header = f"{'名称':<14} {'N':>7} {'D':>6} {'K':>4} {'D/N':>7}  生物学背景"
    print(header)
    print('-' * 80)
    for s in summary:
        print(f"{s['name']:<14} {s['N']:>7} {s['D']:>6} "
              f"{s['K']:>4} {s['D/N']:>7}  {s['context']}")
    print('=' * 80)

    return summary


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='生物医学模拟数据集生成器',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', type=str, default='all',
        help="数据集配置名（如 'SYN-BIO-1'），或 'all' 生成全部"
    )
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--visualize', action='store_true',
        help='生成后自动可视化（低维用 PCA，高维用 t-SNE）'
    )
    args = parser.parse_args()

    if args.config == 'all':
        generate_all_datasets(output_dir=args.output_dir, seed=args.seed)
    else:
        gen = BioDataGenerator(config_name=args.config, seed=args.seed)
        gen.save(output_dir=args.output_dir)
        if args.visualize:
            # 低维用 PCA（快），高维用 t-SNE（慢但效果好）
            method = 'pca' if gen.X.shape[1] < 100 else 'tsne'
            gen.visualize(method=method)
