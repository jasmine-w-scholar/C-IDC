#!/usr/bin/env python
"""
SEFS 特征子集判别力验证

验证：SEFS 选出的少量幸存特征（gates 高的）是否具有判别力？
方法：用幸存特征单独做 K-Means 聚类，看 ACC 是否仍接近全特征水平。

对比三组：
1. 全特征（7129 个）→ baseline ACC
2. SEFS 幸存特征（gates > 阈值）→ 幸存特征 ACC
3. 随机特征（同数量）→ 对照组 ACC

如果幸存特征 ACC 明显高于随机特征，说明 SEFS 选出的特征有判别力。

用法:
    python sefs_feature_validation.py
"""
import numpy as np
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    """匈牙利算法最优匹配，计算聚类 ACC"""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / len(y_true)


def kmeans_acc(X, n_clusters=2, seed=0):
    """对特征矩阵 X 做 KMeans，返回 ACC"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    y_pred = kmeans.fit_predict(X)
    return y_pred


def main():
    # ── 加载数据 ──
    d = np.load('dataset/ALLAML.npz', allow_pickle=True)
    X = StandardScaler().fit_transform(d['X'])
    Y = d['Y'].reshape(-1)
    Y = Y - Y.min()  # 标签从 0 开始
    print(f"数据: {X.shape}, 标签分布: {dict(zip(*np.unique(Y, return_counts=True)))}")

    g_sefs = np.load('gates_ALLAML_sefs.npy')
    g_idc = np.load('gates_ALLAML_idc.npy')

    # ── 1. 全特征 baseline ──
    y_pred_full = kmeans_acc(X)
    acc_full = cluster_acc(Y, y_pred_full)
    print(f"\n{'='*60}")
    print(f"[1] 全特征 (7129) K-Means ACC: {acc_full:.4f}")
    print(f"{'='*60}")

    # ── 2. 对多个 gates 阈值，验证幸存特征判别力 ──
    thresholds = [0.05, 0.1, 0.2, 0.3]
    print(f"\n{'阈值':<8}{'幸存数':<8}{'SEFS幸存ACC':<14}{'随机特征ACC':<14}{'IDC幸存ACC':<14}")
    print("-" * 60)

    for thr in thresholds:
        # SEFS 幸存特征
        surv_sefs = np.where(g_sefs > thr)[0]
        # IDC 幸存特征
        surv_idc = np.where(g_idc > thr)[0]
        # 随机特征（用 SEFS 幸存数做对照组）
        n_surv = len(surv_sefs)
        rng = np.random.RandomState(42)
        rand_idx = rng.choice(X.shape[1], n_surv, replace=False)

        # 计算各组 ACC
        acc_surv = cluster_acc(Y, kmeans_acc(X[:, surv_sefs]))
        acc_rand = cluster_acc(Y, kmeans_acc(X[:, rand_idx]))
        acc_idc = cluster_acc(Y, kmeans_acc(X[:, surv_idc])) if len(surv_idc) > 1 else 0.0

        print(f"{thr:<8.2f}{len(surv_sefs):<8}{acc_surv:<14.4f}{acc_rand:<14.4f}{acc_idc:<14.4f}")

    # ── 3. 多次随机对照（稳健性）──
    print(f"\n{'='*60}")
    print("稳健性验证：SEFS 幸存(311) vs 随机(311)，重复 20 次随机抽样")
    print(f"{'='*60}")
    surv_311 = np.where(g_sefs > 0.1)[0]
    acc_sefs_311 = cluster_acc(Y, kmeans_acc(X[:, surv_311]))
    rng = np.random.RandomState(0)
    acc_rand_list = []
    for i in range(20):
        rand_idx = rng.choice(X.shape[1], len(surv_311), replace=False)
        acc_rand_list.append(cluster_acc(Y, kmeans_acc(X[:, rand_idx])))
    acc_rand_list = np.array(acc_rand_list)
    print(f"SEFS 幸存(311) ACC: {acc_sefs_311:.4f}")
    print(f"随机(311) ACC: 均值={acc_rand_list.mean():.4f}, "
          f"std={acc_rand_list.std():.4f}, 范围=[{acc_rand_list.min():.4f}, {acc_rand_list.max():.4f}]")
    print(f"\nSEFS 幸存特征是否显著优于随机: "
          f"{'✅ 是' if acc_sefs_311 > acc_rand_list.max() else ('⚠️ 部分优于' if acc_sefs_311 > acc_rand_list.mean() else '❌ 否')}")


if __name__ == '__main__':
    main()
