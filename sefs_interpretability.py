#!/usr/bin/env python
"""
SEFS 机制可解释性分析：验证「相关特征同步选择」

核心论证：SEFS 通过 Cholesky 相关噪声，让相关特征（|corr| 高的特征对）
在特征选择时「同步竞争」——同步被选中或同步被淘汰。这是 IDC（独立噪声）
做不到的机制性贡献，与 ACC 提升无关。

实验：
1. 计算 OAS 相关矩阵，找出相关特征对（|corr| > threshold）
2. 训练 SEFS 版（use_sefs=true）和 IDC 版（use_sefs=false）
3. 提取两者的特征 gates（每个特征被选中的程度 0~1）
4. 对比相关特征对的 gates 差值 |g_i - g_j|：
   - SEFS 的相关特征对差值 应该 < IDC（即 SEFS 选择更同步）
   - 同时对比随机特征对作为对照组

用法:
    python sefs_interpretability.py --dataset ALLAML --threshold 0.3 --seed 0
"""
import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
from scipy import stats

from train_evaluate import BaseModule


def compute_corr_abs(data_np):
    """计算 OAS 相关矩阵的绝对值版本（用于找相关特征对），与 dataset.py 逻辑一致"""
    oas = OAS(assume_centered=False)
    oas.fit(data_np)
    cov = oas.covariance_
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-8, 1e-8, std)
    corr = cov / np.outer(std, std)
    corr_abs = np.abs(corr)
    np.fill_diagonal(corr_abs, 0)  # 对角线置 0，排除自相关
    return corr_abs


def train_and_extract_gates(cfg_path, seed=0):
    """训练模型，返回每个特征的平均 gates [D] 和 best_acc"""
    cfg = OmegaConf.load(cfg_path)
    cfg.seed = seed
    cfg.trainer.logger = False  # 关闭日志
    cfg.trainer.enable_checkpointing = False
    seed_everything(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"\n⏳ 训练中: {cfg_path} (seed={seed}) ...")
    model = BaseModule(cfg)
    trainer = Trainer(**cfg.trainer)
    trainer.fit(model)

    # 提取 gates：对全量数据前向传播，得到每个特征的确定性 gates
    x = model.train_dataset.data  # [N, D] StandardScaler 后
    x_tensor = torch.tensor(x).float().to(model.device)
    model.eval()
    with torch.no_grad():
        gates = model.gating_net.get_gates(x_tensor)  # [N, D]
    mean_gates = gates.mean(dim=0).cpu().numpy()  # [D] 每个特征的平均选中程度

    print(f"  ✅ best_acc={model.best_acc:.4f}")
    # 释放显存
    del model, trainer
    torch.cuda.empty_cache()
    return mean_gates


def main():
    parser = argparse.ArgumentParser(description='SEFS 相关特征同步选择分析')
    parser.add_argument('--dataset', default='ALLAML', choices=['ALLAML', 'SRBCT', 'Prostate'])
    parser.add_argument('--threshold', type=float, default=0.3, help='相关特征对阈值 |corr|>threshold')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ds = args.dataset
    data_file = f'dataset/{ds}.npz'
    cfg_sefs = f'cfg/{ds}_best.yaml'
    cfg_idc = f'cfg/{ds}_best_nosefs.yaml'

    # 检查配置文件是否存在
    for f in (data_file, cfg_sefs, cfg_idc):
        if not os.path.exists(f):
            print(f"❌ 文件不存在: {f}")
            return

    # ── 1. 加载数据 + 相关性矩阵 ──
    print(f"加载数据: {data_file}")
    d = np.load(data_file, allow_pickle=True)
    X = StandardScaler().fit_transform(d['X'])
    print(f"数据形状: {X.shape}")
    print(f"计算 OAS 相关矩阵 ...")
    corr_abs = compute_corr_abs(X)

    # ── 2. 找相关特征对 + 随机对照组 ──
    corr_pairs = np.argwhere(corr_abs > args.threshold)
    corr_pairs = corr_pairs[corr_pairs[:, 0] < corr_pairs[:, 1]]  # 去重，只保留 i<j
    n_pairs = len(corr_pairs)
    print(f"相关特征对 (|corr|>{args.threshold}): {n_pairs} 对")

    # 随机对照组（同数量的随机特征对）
    D = X.shape[1]
    rng = np.random.RandomState(42)
    rand_i = rng.randint(0, D, n_pairs)
    rand_j = rng.randint(0, D, n_pairs)
    rand_pairs = np.stack([rand_i, rand_j], axis=1)

    # ── 3. 训练两个版本 + 提取 gates ──
    gates_sefs = train_and_extract_gates(cfg_sefs, seed=args.seed)
    gates_idc = train_and_extract_gates(cfg_idc, seed=args.seed)

    # 保存 gates 供后续分析（生物学意义等）
    np.save(f'gates_{ds}_sefs.npy', gates_sefs)
    np.save(f'gates_{ds}_idc.npy', gates_idc)
    print(f"\n💾 gates 已保存: gates_{ds}_sefs.npy, gates_{ds}_idc.npy")

    # ── 4. 分析同步选择 ──
    def diff_gates(gates, pairs):
        return np.abs(gates[pairs[:, 0]] - gates[pairs[:, 1]])

    diff_sefs_corr = diff_gates(gates_sefs, corr_pairs)
    diff_idc_corr = diff_gates(gates_idc, corr_pairs)
    diff_sefs_rand = diff_gates(gates_sefs, rand_pairs)
    diff_idc_rand = diff_gates(gates_idc, rand_pairs)

    print("\n" + "=" * 60)
    print("SEFS 机制可解释性分析：相关特征同步选择")
    print("=" * 60)
    print(f"相关特征对 gates 差值均值 |g_i - g_j|（越小越同步）:")
    print(f"  SEFS: {diff_sefs_corr.mean():.4f}")
    print(f"  IDC : {diff_idc_corr.mean():.4f}")
    print(f"随机特征对 gates 差值均值（对照组）:")
    print(f"  SEFS: {diff_sefs_rand.mean():.4f}")
    print(f"  IDC : {diff_idc_rand.mean():.4f}")

    # t 检验：SEFS 相关 vs IDC 相关
    t_corr, p_corr = stats.ttest_ind(diff_sefs_corr, diff_idc_corr, equal_var=False)
    print(f"\n[SEFS vs IDC 相关特征对] Welch t={t_corr:.4f}, p={p_corr:.4f}")

    # 同步选择强度：随机差值均值 - 相关差值均值（越大说明相关特征越同步）
    sync_sefs = diff_sefs_rand.mean() - diff_sefs_corr.mean()
    sync_idc = diff_idc_rand.mean() - diff_idc_corr.mean()
    print(f"\n同步选择强度（随机差值 - 相关差值，越大越同步）:")
    print(f"  SEFS: {sync_sefs:.4f}")
    print(f"  IDC : {sync_idc:.4f}")

    # 结论
    print("\n" + "=" * 60)
    if sync_sefs > sync_idc and p_corr < 0.05:
        print("✅ 验证成立：SEFS 的相关特征选择显著更同步")
    elif sync_sefs > sync_idc:
        print("⚠️ 方向正确（SEFS 更同步），但统计不显著")
    else:
        print("❌ 未验证到 SEFS 同步选择的优势")


if __name__ == '__main__':
    main()
