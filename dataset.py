from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.covariance import OAS
from sklearn.cluster import AgglomerativeClustering
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn import datasets


class ClusteringDataset(Dataset):
    def __init__(self, data, labels=None, num_clusters=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self._num_clusters = num_clusters
        if num_clusters is None and labels is None:
            raise ValueError("At least one of the values should be provided (labels/num_clusters)")
        self.print_stats()

    def __getitem__(self, index: int):
        if self.labels is None:
            return torch.tensor(self.data[index]).float()
        return torch.tensor(self.data[index]).float(), torch.tensor(self.labels[index]).long()

    def __len__(self) -> int:
        return len(self.data)

    @property
    def num_clusters(self):
        return self._num_clusters if self._num_clusters is not None else len(np.unique(self.labels))

    def num_features(self):
        return self.data.shape[-1]

    def print_stats(self):
        print('X.shape: ', self.data.shape)
        print(f"X.min={self.data.min()}, X.max={self.data.max()}")
        if self.labels is not None:
            print('Y.shape: ', self.labels.shape)
            for y_u in np.unique(self.labels):
                print(f'{y_u}: {np.sum(self.labels == y_u)}')
            print(f"Y.min={self.labels.min()}, Y.max={self.labels.max()}")

    @classmethod
    def setup(cls, cfg):
        pass


class NumpyTableDataset(ClusteringDataset):

    def __init__(self, data, labels=None, num_clusters=None,correlation_threshold = 0.3):
        # ── IDC 原版逻辑（完全保留）──
        # super().__init__ 完成后 self.data 才被正确赋值
        super().__init__(data, labels, num_clusters)

         # ════════════════════════════════════════════════════════
        # ✅[新增]SEFS 预计算：特征相关性矩阵 + Cholesky 分解
        #
        # 必须在 super().__init__() 之后执行，原因：
        #   super().__init__() 中完成 self.data = data 的赋值，
        #   Cholesky 分解依赖 self.data，故必须在此之后。
        #
        # 对应 SEFS 论文 Algorithm 1/3：
        #   R_kj = |C_kj| / sqrt(C_kk * C_jj)  （皮尔逊相关系数绝对值）
        #   L    = Cholesky(R)                   （下三角分解，R = L @ L^T）
        # ════════════════════════════════════════════════════════
        self.correlation_matrix = self._compute_correlation_matrix()

        self.cluster_labels = self._compute_feature_clusters(
            correlation_threshold=correlation_threshold
        )
        # self.cholesky_L = self._compute_cholesky_L()
        self.cholesky_L = self._compute_block_cholesky_L(
            cluster_labels=self.cluster_labels,
            min_cluster_size=2
        )
        corr_np = self.correlation_matrix.numpy()
        np.fill_diagonal(corr_np, 0)
        max_off_diag = corr_np.max()
        print(f"✅ [SEFS 诊断] 最大特征间相关性（非对角）: {max_off_diag:.4f}")
        if max_off_diag < 0.3:
            print("⚠️ [SEFS 诊断] 特征间相关性较低，SEFS 改进效果可能有限")
        elif max_off_diag > 0.7:
            print("✅ [SEFS 诊断] 特征间存在强相关，SEFS 将有效抑制冗余特征选择")

        # ═══════════════════════════════════════════════════════════
        # ✅ 内存释放：Cholesky 完成后，以下大矩阵不再需要。
        #   7129×7129 的矩阵每个约 203MB(float32) / 406MB(float64)，
        #   及时释放可显著降低 CPU 内存峰值，避免多进程并发时内存溢出。
        # ═══════════════════════════════════════════════════════════
        for _attr in ('_corr_for_cholesky', '_corr_for_clustering',
                      'correlation_matrix', 'cluster_labels'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        import gc
        gc.collect()

    def _compute_correlation_matrix(self) -> torch.Tensor:
        """
        [修复版] 使用 OAS 估计量计算正则化相关矩阵。

        核心修复：
            将协方差矩阵先归一化为含正负值的 Pearson 相关矩阵，
            再取绝对值，确保绝对值操作不破坏 OAS 的正定性保证。

        修复前（错误）：corr_oas = np.abs(cov_oas) / outer_std
            → 对协方差矩阵直接取绝对值再归一化，破坏正定性
            → 最小特征值可能为 -5.58（SRBCT 实测）

        修复后（正确）：
            Step A: corr_signed = cov_oas / outer_std  （先归一化，保持正定性）
            Step B: corr_oas    = np.abs(corr_signed)  （再取绝对值）
            → 归一化是行列缩放，不改变正半定性
            → 取绝对值在已归一化的相关矩阵上操作，数值范围 [-1,1] → [0,1]

        同时维护两个矩阵供后续使用：
            self._corr_for_cholesky   : 含正负值相关矩阵（用于 Cholesky 分解）
            self._corr_for_clustering : 取绝对值相关矩阵（用于特征聚类距离计算）

        返回：torch.Tensor [D, D]，值域 [0, 1]（绝对值版本，供诊断/聚类使用）
        """
        data_np = self.data  # np.ndarray [N, D]
        n_samples, n_features = data_np.shape
        print(f"[OAS] 数据维度: n={n_samples}, D={n_features}, "
              f"D/n比={n_features / n_samples:.2f}")

        # ── Step 1: OAS 拟合 ────────────────────────────────────────
        oas = OAS(assume_centered=False)
        oas.fit(data_np)
        cov_oas = oas.covariance_  # np.ndarray [D, D]，OAS 正则化协方差矩阵
        shrinkage = oas.shrinkage_
        print(f"[OAS] 收缩强度 α_OAS = {shrinkage:.4f}")

        if shrinkage > 0.9:
            print(f"⚠️ [OAS] α_OAS={shrinkage:.4f} 仍然较高，"
                  f"D/n={n_features / n_samples:.1f} 极端，"
                  f"但已低于同数据下的 LW 收缩强度")
        elif shrinkage > 0.7:
            print(f"⚠️ [OAS] α_OAS={shrinkage:.4f}，相关结构有所压缩，"
                  f"SEFS 竞争信号中等")
        else:
            print(f"✅ [OAS] α_OAS={shrinkage:.4f}，相关结构良好保留，"
                  f"SEFS 竞争信号较强")

            # ── Step 2: 协方差矩阵 → Pearson 相关矩阵（含正负值）──────────
        # 关键：先归一化（行列缩放，不改变正定性），再取绝对值
        std_vec = np.sqrt(np.diag(cov_oas))  # [D]
        std_vec = np.where(std_vec < 1e-8, 1e-8, std_vec)  # 防止零方差特征除零
        outer_std = np.outer(std_vec, std_vec)  # [D, D]

        # ✅ 修复核心：先归一化为含正负值的相关矩阵
        corr_signed = cov_oas / outer_std  # [D, D]，值域 [-1, 1]
        np.fill_diagonal(corr_signed, 1.0)  # 对角线强制为 1.0
        corr_signed = np.clip(corr_signed, -1.0, 1.0)  # 裁剪至 [-1, 1]

        # ✅ 修复核心：再对已归一化的相关矩阵取绝对值（用于聚类）
        corr_abs = np.abs(corr_signed)  # [D, D]，值域 [0, 1]
        np.fill_diagonal(corr_abs, 1.0)
        corr_abs = np.clip(corr_abs, 0.0, 1.0)

        # ── Step 3: 分别保存两个矩阵供后续使用 ─────────────────────────
        # corr_for_cholesky   : 含正负值，用于 Cholesky 分解（保持正定性）
        # corr_for_clustering : 取绝对值，用于特征聚类距离计算
        self._corr_for_cholesky = corr_signed  # np.ndarray，含正负值
        self._corr_for_clustering = corr_abs  # np.ndarray，取绝对值

        # ── Step 4: 诊断输出 ────────────────────────────────────────
        corr_diag0 = corr_abs.copy()
        np.fill_diagonal(corr_diag0, 0)
        max_off_diag = corr_diag0.max()
        mean_off_diag = (corr_diag0.sum() / (n_features * (n_features - 1))
                         if n_features > 1 else 0.0)
        print(f"[OAS] 最大非对角相关性（OAS正则化后）: {max_off_diag:.4f}")
        print(f"[OAS] 平均非对角相关性（OAS正则化后）: {mean_off_diag:.6f}")

        if max_off_diag < 0.3:
            print("⚠️ [OAS] 特征间相关性较低，SEFS 改进效果可能有限")
        elif max_off_diag > 0.7:
            print("✅ [OAS] 特征间存在强相关，SEFS 将有效抑制冗余特征选择")

            # ── Step 5: 正定性验证（基于用于 Cholesky 的含正负值矩阵）──────
        # 注意：此处验证 corr_signed（Cholesky 的实际输入），而非 corr_abs
        eigvals = np.linalg.eigvalsh(corr_signed)
        min_eigval = eigvals.min()
        print(f"[OAS] 相关矩阵最小特征值（Cholesky输入矩阵）: {min_eigval:.6e}")
        if min_eigval <= 0:
            print(f"⚠️ [OAS] 最小特征值 {min_eigval:.6e} <= 0，"
                  f"OAS 正定性保证失效（请检查数据预处理，"
                  f"是否存在完全零方差的特征列）")
        else:
            print(f"✅ [OAS] Cholesky输入相关矩阵严格正定（最小特征值 {min_eigval:.6e} > 0）")

            # ── Step 6: 返回绝对值版本（与原版接口一致，供聚类/诊断使用）────
        corr_tensor = torch.from_numpy(corr_abs).float()
        print(f"✅ [OAS] OAS 正则化相关矩阵计算完成，形状: {corr_tensor.shape}")
        return corr_tensor

    def _compute_feature_clusters(
            self,
            correlation_threshold: float = 0.3
    ) -> np.ndarray:
        """
        参数：
            correlation_threshold: float
                特征对相关性阈值，高于此值的特征对归入同一簇。
                建议根据数据集实际最大相关性调整：
        """
        # ── 使用绝对值相关矩阵构造距离（专用于聚类）────────────────────
        # 优先使用 _corr_for_clustering（修复后新增属性），
        # 兜底使用 self.correlation_matrix（两者内容一致，保证向后兼容）
        if hasattr(self, '_corr_for_clustering'):
            corr_np = self._corr_for_clustering  # np.ndarray [D, D]，值域 [0,1]
        else:
            corr_np = self.correlation_matrix.numpy()  # 向后兼容

        distances = 1.0 - corr_np  # [D, D]，相关距离
        np.fill_diagonal(distances, 0.0)
        distances = np.clip(distances, 0.0, 1.0)

        # ── 凝聚聚类 ────────────────────────────────────────────────
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='complete',
            distance_threshold=(1.0 - correlation_threshold)
        )
        cluster_labels = clustering.fit_predict(distances)  # [D] int array

        # ── 诊断输出 ─────────────────────────────────────────────────
        D = len(cluster_labels)
        K = len(np.unique(cluster_labels))
        sizes = [int(np.sum(cluster_labels == c)) for c in np.unique(cluster_labels)]
        max_size = max(sizes)
        min_size = min(sizes)
        n_singleton = sum(1 for s in sizes if s == 1)
        n_samples = self.data.shape[0]

        print(f"[块级Cholesky] 特征分簇结果（OAS相关矩阵，阈值={correlation_threshold}）：")
        print(f"   总特征数 D       = {D}")
        print(f"   识别簇数 K       = {K}")
        print(f"   最大簇大小       = {max_size} (D_g/n = {max_size / n_samples:.2f})")
        print(f"   最小簇大小       = {min_size}")
        print(f"   孤立特征（单特征簇）= {n_singleton} 个")
        print(f"   SEFS 覆盖率（非孤立特征比例）= {(D - n_singleton) / D * 100:.1f}%")

        if n_singleton > D * 0.9:
            print(f"⚠️ [块级Cholesky] 超过90%特征为孤立簇，OAS 收缩仍较强，"
                  f"SEFS 相关机制对大多数特征退化为独立噪声")
        elif n_singleton < D * 0.5:
            print(f"✅ [块级Cholesky] 超过50%特征参与相关簇，"
                  f"OAS 有效保留了相关结构，SEFS 竞争机制较强")

        if max_size > n_samples:
            print(f"⚠️ [块级Cholesky] 最大簇大小 {max_size} > n({n_samples})，"
                  f"该块依赖 OAS 正则化保证正定性（不影响 Cholesky 成功）")

        return cluster_labels  # np.ndarray [D], dtype=int

    def _compute_block_cholesky_L(
            self,
            cluster_labels: np.ndarray,
            min_cluster_size: int = 2
    ) -> torch.Tensor:
        """
        [修复版] 对每个特征簇的 OAS 子矩阵分别做 Cholesky 分解。

        核心修复：
            Cholesky 分解的子矩阵来源从 corr_abs（绝对值相关矩阵）
            改为 self._corr_for_cholesky（含正负值、严格正定的相关矩阵）。

            修复前：R_sub = corr_abs[ix_(indices, indices)]
                → corr_abs 取了绝对值，破坏了正定性（负特征值 -5.58）
                → 依赖 jitter 强行分解，数学不自洽

            修复后：R_sub = corr_signed[ix_(indices, indices)]
                → corr_signed 是归一化后未取绝对值的相关矩阵，严格正定
                → jitter=1e-8 理论上第一次即成功，无需大量 jitter 扰动
        """
        D = len(cluster_labels)
        L_block = np.zeros([D, D], dtype=np.float64)

        # ── 修复核心：Cholesky 使用含正负值的正定相关矩阵 ──────────────
        if hasattr(self, '_corr_for_cholesky'):
            corr_np = self._corr_for_cholesky.astype(np.float64)  # 含正负值，严格正定
        else:
            # 向后兼容：若未调用修复版 _compute_correlation_matrix，
            # 退回使用 self.correlation_matrix（绝对值版本，可能不正定）
            print("⚠️ [块级Cholesky] 未找到 _corr_for_cholesky，"
                  "退回使用 correlation_matrix（可能含负特征值）")
            corr_np = self.correlation_matrix.numpy().astype(np.float64)

        unique_c = np.unique(cluster_labels)
        success_count = 0
        fallback_count = 0
        skip_count = 0

        for cluster_id in unique_c:
            indices = np.where(cluster_labels == cluster_id)[0]
            n_sub = len(indices)

            # ── 情况A：孤立特征（n_sub < min_cluster_size）──────────────
            # 孤立特征使用独立标准高斯，L_ii = 1.0
            if n_sub < min_cluster_size:
                for idx in indices:
                    L_block[idx, idx] = 1.0
                skip_count += 1
                continue

                # ── 情况B：有效簇，提取正定子相关矩阵做 Cholesky ──────────────
            # ✅ 修复：使用 corr_signed（含正负值，正定）而非 corr_abs（可能不正定）
            R_sub = corr_np[np.ix_(indices, indices)]  # [n_sub, n_sub]，严格正定
            jitter = 1e-8
            success = False

            for attempt in range(6):
                try:
                    R_stable = R_sub + jitter * np.eye(n_sub)
                    L_sub = np.linalg.cholesky(R_stable)  # 下三角矩阵
                    success = True
                    success_count += 1
                    if attempt > 0:
                        print(f"  [块级Cholesky] 簇{cluster_id}（大小{n_sub}）"
                              f"在 jitter={jitter / 10:.0e} 时成功（尝试第{attempt + 1}次）")
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10  # 1e-8 → 1e-7 → ... → 1e-3

            if not success:
                L_sub = np.eye(n_sub)
                fallback_count += 1
                print(f"⚠️ [块级Cholesky] 簇{cluster_id}（大小{n_sub}）"
                      f"6次 Cholesky 均失败（jitter 最大至 1e-3），"
                      f"退化为单位阵（请检查数据预处理是否存在常数特征列）")

            L_block[np.ix_(indices, indices)] = L_sub

            # ── 诊断输出 ─────────────────────────────────────────────────
        n_nonzero_diag = int(np.sum(np.abs(np.diag(L_block)) > 1e-10))
        total_valid = len(unique_c) - skip_count

        print(f"[块级Cholesky] OAS + 块级 Cholesky 分解完成：")
        print(f"   有效簇（Cholesky成功）= {success_count}/{total_valid}")
        print(f"   退化簇（单位阵兜底）  = {fallback_count}（应为0）")
        print(f"   孤立特征簇（单位对角）= {skip_count}")
        print(f"   L_block 非零对角元素  = {n_nonzero_diag}/{D}")

        if fallback_count > 0:
            print(f"⚠️ 出现 {fallback_count} 个退化簇，"
                  f"请检查 OAS 是否正确拟合，或数据中是否存在常数特征列")
        if n_nonzero_diag < D:
            print(f"⚠️ 有 {D - n_nonzero_diag} 个特征对角线为0，"
                  f"这些特征在 GatingNet 中产生零噪声，门控退化")

            # ── 数学验证 ─────────────────────────────────────────────────
        LLT = L_block @ L_block.T
        diag_vals = np.diag(LLT)
        n_valid_diag = int(np.sum(np.abs(diag_vals - 1.0) < 0.01))
        print(f"[数学验证] L_block @ L_block^T 对角线验证：")
        print(f"   对角线接近1.0的元素数 = {n_valid_diag}/{D}")
        if n_valid_diag < D:
            bad_indices = np.where(np.abs(diag_vals - 1.0) >= 0.01)[0]
            print(f"⚠️ 对角线偏差超过0.01的特征索引（前5个）: "
                  f"{bad_indices[:5].tolist()}")

        L_block_tensor = torch.from_numpy(L_block).float()
        print(f"✅ [块级Cholesky] L_block 计算完成，形状: {L_block_tensor.shape}")
        return L_block_tensor
        return L_block_tensor

    def _compute_cholesky_L(self):
        """
        [保留原版，已被 _compute_block_cholesky_L 替代]
        对相关性矩阵 R 做 Cholesky 分解，得到下三角矩阵 L

        保留此方法作为回退选项，便于对比实验：
            self.cholesky_L = self._compute_cholesky_L()      # 原版
            self.cholesky_L = self._compute_block_cholesky_L(...) # OAS版
        """
        R = self.correlation_matrix.clone()  # torch.Tensor [D, D]
        D = R.size(0)
        jitter = 1e-5

        for attempt in range(10):
            try:
                R_stable = R + torch.eye(D) * jitter
                L = torch.linalg.cholesky(R_stable)
                print(f"✅ [SEFS] Cholesky 分解成功 "
                      f"(jitter={jitter:.2e}, L.shape={L.shape})")
                return L
            except RuntimeError:
                jitter *= 10
                print(f"⚠️ [SEFS] Cholesky 分解失败，增大 jitter 至 {jitter:.2e} 重试...")

        print("⚠️ [SEFS] 警告：Cholesky 分解彻底失败，退化为单位阵")
        return torch.eye(D)
    @classmethod
    def setup(cls,
              filepath_samples: str,
              filepath_labels: str = None,correlation_threshold: float = 0.3,
              num_clusters: int = None):
        """
        加载数据的工厂方法

        相对于 IDC GitHub 原版（仅支持 'arr_0' 键名）的修复：
        ✅ 修复1：支持多种键名（按优先级顺序尝试）：
               'X'     ← correlate_data.npz / test_data.npz 格式
               'data'  ← 其他标准格式
               'arr_0' ← IDC 原版 / np.savez 默认无名格式
        ✅ 修复2：标签加载支持从同一文件读取 'Y' / 'labels' / 'arr_0' 键
        ✅ 修复3：StandardScaler 在有标签和无标签情况下均执行
               （原版仅在有 filepath_labels 时执行，逻辑有误）
        ✅ 保持：返回 cls(X, Y, num_clusters)，与 IDC 原版接口完全一致

        参数：
            filepath_samples: .npz 文件路径（数据矩阵 [N, D]）
            filepath_labels:  .npz 文件路径（标签向量 [N]，可选）
            num_clusters:     簇数量（可选，None 时从标签自动推断）
        """

        # ── Step 1：加载数据文件 ─────────────────────────────
        data_dict = np.load(filepath_samples, allow_pickle=True)

        # ✅[修复核心]多键名兼容，按优先级顺序尝试
        # correlate_data.py 保存时用的是 np.savez(filename, X=self.X, Y=self.Y)
        # 所以 correlate_data.npz 的键名是 'X'，而非 IDC 原版的 'arr_0'
        if 'X' in data_dict:
            X = data_dict['X']  # ← correlate_data.npz / test_data.npz 格式
            print(f"✅ [Dataset] 使用键名 'X' 加载数据，形状: {X.shape}")
        elif 'data' in data_dict:
            X = data_dict['data']  # ← 其他标准格式
            print(f"✅ [Dataset] 使用键名 'data' 加载数据，形状: {X.shape}")
        elif 'arr_0' in data_dict:
            X = data_dict['arr_0']  # ← IDC 原版 / np.savez 默认无名格式
            print(f"✅ [Dataset] 使用键名 'arr_0' 加载数据，形状: {X.shape}")
        else:
            raise KeyError(
                f"在 {filepath_samples} 中找不到数据。\n"
                f"可用键名: {list(data_dict.keys())}\n"
                f"请确保 .npz 文件包含 'X'、'data' 或 'arr_0' 键之一。\n"
                f"提示：correlate_data.py 生成的文件使用 'X' 键名。"
            )

            # ── Step 2：加载标签（可选）────────────────────────────
        Y = None
        if filepath_labels is not None:
            # 从独立的标签文件加载
            with np.load(filepath_labels) as label_data:
                if 'Y' in label_data:
                    Y = label_data['Y']
                elif 'arr_0' in label_data:
                    Y = label_data['arr_0']  # IDC 原版标签文件格式
                else:
                    print(f"⚠️ [Dataset] 标签文件 {filepath_labels} 中未找到 'Y' 或 'arr_0'，"
                          f"可用键: {list(label_data.keys())}，标签设为 None")
        else:
            # 尝试从同一数据文件中读取标签
            if 'Y' in data_dict:
                Y = data_dict['Y']  # ← correlate_data.npz 同时存有 Y
                print(f"✅ [Dataset] 从同一文件中读取标签 'Y'，形状: {Y.shape}")
            elif 'labels' in data_dict:
                Y = data_dict['labels']
                print(f"✅ [Dataset] 从同一文件中读取标签 'labels'，形状: {Y.shape}")
                # 若均未找到，Y 保持 None（无监督聚类场景允许无标签）

        # ── Step 3：数据标准化（StandardScaler）────────────────
        # ✅[修复]原版仅在 filepath_labels is not None 时执行 StandardScaler，
        #   但无论是否有标签文件，都应该对 X 做标准化（神经网络训练对尺度敏感）
        X = preprocessing.StandardScaler().fit_transform(X)
        # x_min, x_max = X.min(), X.max()
        # print(f"✅ [Dataset] 加载后数据值域: [{x_min:.4f}, {x_max:.4f}]")
        #
        # if x_min >= 0.0 and x_max <= 1.0:
        #     # ✅ correlate_data 生成时已做 MinMaxScaler → 直接使用
        #     print("✅ [Dataset] 检测到 [0,1] 范围数据，跳过 StandardScaler")
        # else:
        #     # ✅ 其他未归一化数据 → 执行 StandardScaler
        #     X = preprocessing.StandardScaler().fit_transform(X)
        #     print("✅ [Dataset] 已执行 StandardScaler")
            # ── Step 4：标签后处理 ──────────────────────────────────
        if Y is not None:
            Y = Y - Y.min()  # 确保标签从 0 开始（与 IDC 原版一致）
            if num_clusters is None:
                num_clusters = len(np.unique(Y))
        return cls(X, Y, num_clusters)

def remove_zero_columns(X):
    non_zero_columns = []
    for col in range(X.shape[1]):
        if np.min(X[:, col]) == 0 and np.max(X[:, col]) == 0:
            continue
        else:
            non_zero_columns.append(col)
    X = X[:, non_zero_columns]
    return X

class PBMC(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        data_dir = cfg.data_dir
        with np.load(f"{data_dir}/pbmc_x.npz") as data:
            X = data['arr_0']
        with np.load(f"{data_dir}/pbmc_y.npz") as data:
            Y = data['arr_0']
        Y = Y - Y.min()
        scaler = getattr(preprocessing, cfg.scaler)()
        X = scaler.fit_transform(X)
        return cls(X, Y)


class MNIST10K(ClusteringDataset):
    def __init__(self, data, targets):
        super().__init__(data, targets)

    @classmethod
    def setup(cls, cfg):
        scaler = getattr(preprocessing, cfg.scaler)()
        X = MNIST(cfg.data_dir, train=True, download=True).data.reshape(-1, 784).cpu().numpy()
        Y = MNIST(cfg.data_dir, train=True, download=True).targets.cpu().numpy()
        X = scaler.fit_transform(X)
        X = X[:10000]
        Y = Y[:10000]
        return cls(X, Y)