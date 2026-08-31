# ============================================================
# IDC+SEFS 贝叶斯超参数优化脚本
# 文件名：sefs_bayes_opt.py
#
# 使用方式：
#   python sefs_bayes_opt.py  --cfg cfg/test.yaml  --n_calls 20  --batch_size 4  --n_workers 3 --n_seeds 10  --early_stop --patience 20  --min_delta 0.005
#   python sefs_bayes_opt.py  --cfg cfg/SYN-BIO-6.yaml  --n_calls 40  --batch_size 4  --n_workers 4  --n_seeds 10  --early_stop  --patience 40 --min_delta 0.005
# 依赖：
#   pip install scikit-optimize pandas pytorch-lightning
# ============================================================

import os
import json
import math
import argparse
import threading
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from skopt import Optimizer
from skopt.space import Real, Integer
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import gc

# ═══════════════════════════════════════════════════════════════
# ✅ GPU 内存防 OOM 配置
# ═══════════════════════════════════════════════════════════════
# 减少 CUDA 内存碎片，降低大块分配失败概率
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
# 限制每个进程仅使用一张 GPU（多进程场景下关键）
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# ── 从训练脚本中导入模型（确保 train_evaluate.py 在同级目录）──
from train_evaluate import BaseModule

torch.set_float32_matmul_precision('high')


# ================================================================
# ✅ 条件早停：在预训练阶段（ae_pretrain_epochs）结束前不监控指标
#    原因：预训练阶段不记录 val/acc_single，若立即监控会报错
# ================================================================
class ConditionalEarlyStopping(EarlyStopping):
    """
    健壮的条件早停回调

    相比标准 EarlyStopping 的改进：
    1. 在 start_epoch 之前完全跳过检查，避免预训练阶段误触发
    2. 监控指标不存在时静默跳过（不报错），适配 val 频率不整除的情况
    3. 首次发现指标可用时打印提示信息
    """

    def __init__(self, start_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self._first_check = True  # 用于控制首次打印

    def _run_early_stopping_check(self, trainer):
        # ── 条件一：未到预训练结束，直接跳过 ──
        if trainer.current_epoch < self.start_epoch:
            return

        # ── 条件二：监控指标当前不可用，静默跳过 ──
        logs = trainer.callback_metrics
        if self.monitor not in logs:
            if self._first_check:
                print(f"⚠️  [早停] 指标 '{self.monitor}' 暂不可用，跳过检查")
                print(f"   当前可用指标: {list(logs.keys())[:5]}")
                self._first_check = False
            return

        # ── 条件三：正常早停逻辑 ──
        self._first_check = False
        super()._run_early_stopping_check(trainer)


# ================================================================
# ✅ 单次参数评估函数（在子进程中运行，避免 GPU 内存泄漏）
#
# 参数说明：
#   cfg_dict         - 基础配置字典（从 yaml 加载后转换）
#   以下均为待优化的超参数
#   seed             - 当前随机种子
#   early_stop_config - 早停配置字典
# ================================================================
def evaluate_single_params(
        cfg_dict,
        batch_size,
        correlation_threshold,
        # ── 训练周期相关 ──
        epochs,
        ae_non_gated_epochs,
        ae_pretrain_epochs,
        start_global_gates_training_on_epoch,
        # ── 网络架构 ──
        gates_hidden_dim,
        # ── 学习率 ──
        lr_pretrain,
        lr_clustering,
        lr_aux_classifier,
        sched_pretrain_min_lr,
        sched_clustering_min_lr,
        # ── 损失权重 ──
        local_gates_lambda,
        global_gates_lambda,
        gtcr_lambda,
        # ── IDC 聚类参数 ──
        tau,
        eps,
        mask_percentage,
        # ── SEFS 专用参数（核心）──
        sefs_tau,
        # ── 随机种子与早停配置 ──
        seed,
        early_stop_config
):
    """
    评估单组超参数的性能

    返回字典：
        acc            - 最佳聚类准确率
        ari            - 最佳 ARI
        nmi            - 最佳 NMI
        actual_epochs  - 实际训练 epoch 数（早停触发则 < max_epochs）
        early_stopped  - 是否触发早停
        success        - 是否成功完成评估
    """
    try:
        # ── Step 1: 构建配置 ──
        cfg = OmegaConf.create(cfg_dict)
        cfg.batch_size = int(batch_size)  # ← 新增
        cfg.correlation_threshold = float(correlation_threshold)  # ← 新增
        # ── Step 2: 覆写待优化超参数（注意类型转换）──
        cfg.epochs                              = int(epochs)
        cfg.ae_non_gated_epochs                 = int(ae_non_gated_epochs)
        cfg.ae_pretrain_epochs                  = int(ae_pretrain_epochs) 
        cfg.start_global_gates_training_on_epoch = int(start_global_gates_training_on_epoch)
        cfg.gates_hidden_dim                    = int(gates_hidden_dim)
        cfg.lr.pretrain                         = float(lr_pretrain)
        cfg.lr.clustering                       = float(lr_clustering)
        cfg.lr.aux_classifier                   = float(lr_aux_classifier)
        cfg.sched.pretrain_min_lr               = float(sched_pretrain_min_lr)
        cfg.sched.clustering_min_lr             = float(sched_clustering_min_lr)
        cfg.local_gates_lambda                  = float(local_gates_lambda)
        cfg.global_gates_lambda                 = float(global_gates_lambda)
        cfg.gtcr_lambda                         = float(gtcr_lambda)
        cfg.tau                                 = float(tau)
        cfg.eps                                 = float(eps)
        cfg.mask_percentage                     = float(mask_percentage)
        cfg.sefs_tau                            = float(sefs_tau)   # ← SEFS 专用
        cfg.use_sefs                            = True              # ← 固定启用 SEFS
        cfg.trainer.max_epochs                  = int(epochs)
        cfg.seed                                = int(seed)

        # ── 约束检查：epoch 顺序合法性 ──
        # ae_non_gated < ae_pretrain < start_global <= epochs
        if not (cfg.ae_non_gated_epochs < cfg.ae_pretrain_epochs):
            cfg.ae_non_gated_epochs = max(10, cfg.ae_pretrain_epochs - 50)
        if not (cfg.ae_pretrain_epochs < cfg.start_global_gates_training_on_epoch):
            cfg.start_global_gates_training_on_epoch = cfg.ae_pretrain_epochs + 50
        if not (cfg.start_global_gates_training_on_epoch < cfg.epochs):
            cfg.epochs = cfg.start_global_gates_training_on_epoch + 100
            cfg.trainer.max_epochs = cfg.epochs

        # ── Step 3: 设置随机种子 ──
        seed_everything(int(seed))
        np.random.seed(int(seed))

        # ── Step 4: 禁用日志（加速评估，避免文件 I/O 竞争）──
        cfg.trainer.logger = False
        cfg.trainer.enable_checkpointing = False

        # ── Step 5: 配置早停回调 ──
        callbacks = []
        early_stop_callback = None

        if early_stop_config.get('enabled', False):
            # 计算早停启动 epoch（确保至少经历一次验证）
            val_freq = cfg.trainer.get('check_val_every_n_epoch', 10)
            start_epoch = int(ae_pretrain_epochs)

            # 对齐到验证频率的整数倍，确保第一次检查时指标一定存在
            if start_epoch % val_freq != 0:
                start_epoch = ((start_epoch // val_freq) + 1) * val_freq

            early_stop_callback = ConditionalEarlyStopping(
                start_epoch=start_epoch,
                monitor=early_stop_config.get('monitor', 'val/acc_single'),
                patience=early_stop_config.get('patience', 30),
                mode=early_stop_config.get('mode', 'max'),
                min_delta=early_stop_config.get('min_delta', 0.001),
                verbose=early_stop_config.get('verbose', False),
            )
            callbacks.append(early_stop_callback)

        # ── Step 6: 初始化模型并训练 ──
        model   = BaseModule(cfg)
        trainer = Trainer(**cfg.trainer, callbacks=callbacks)
        trainer.fit(model)
        acc = model.best_acc if hasattr(model, 'best_acc') else 0.0
        ari = model.best_ari if hasattr(model, 'best_ari') else 0.0
        nmi = model.best_nmi if hasattr(model, 'best_nmi') else 0.0

        # ✅ 计算 Top-K Silhouette Score
        if hasattr(model, 'max_silhouette_score') and len(model.max_silhouette_score) > 0:
            k = min(10, len(model.max_silhouette_score))
            topk_max_silhouette = np.mean(
                sorted(model.max_silhouette_score, reverse=True)[:k]
            )
        else:
            topk_max_silhouette = 0.1

        # ✅ 计算 Top-K DBI Score（越小越好）
        if hasattr(model, 'min_dbi_score') and len(model.min_dbi_score) > 0:
            k = min(10, len(model.min_dbi_score))
            topk_min_dbi = np.mean(
                sorted(model.min_dbi_score)[:k]
            )
        else:
            topk_min_dbi = 3.0

        # ── Step 7: 多目标综合评分 ──
        multi_objective_score = calculate_multi_objective_score(
            acc=acc, ari=ari, nmi=nmi,
            topk_max_silhouette=topk_max_silhouette,
            topk_min_dbi=topk_min_dbi
        )

        # ── Step 8: 收集早停信息 ──
        actual_epochs = trainer.current_epoch
        early_stopped = (
                early_stop_callback is not None and
                early_stop_callback.stopped_epoch > 0
        )

        print(f"   🎯 多目标评分: {multi_objective_score:.4f} "
              f"(ACC={acc:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}, "
              f"Sil={topk_max_silhouette:.3f}, DBI={topk_min_dbi:.3f})")

        # ── ✅ 立即清理显存（在 return 之前，确保释放）──
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
            torch.cuda.empty_cache()
        gc.collect()

        return {
            'acc': acc,
            'ari': ari,
            'nmi': nmi,
            'topk_max_silhouette': topk_max_silhouette,
            'topk_min_dbi': topk_min_dbi,
            'multi_objective_score': multi_objective_score,
            'actual_epochs': actual_epochs,
            'early_stopped': early_stopped,
            'success': True,
        }

    except Exception as e:
        print(f"❌ [评估失败] Seed={seed}, 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        # ✅ 失败时报告 GPU 内存状态，帮助诊断 OOM
        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                print(f"   📊 GPU 内存: {free/1024**3:.1f}GB 空闲 / {total/1024**3:.1f}GB 总计")
            except:
                pass
        return {
            'acc': 0.0, 'ari': 0.0, 'nmi': 0.0,
            'topk_max_silhouette': 0.0, 'topk_min_dbi': 5.0,
            'multi_objective_score': 0.0,
            'actual_epochs': 0, 'early_stopped': False, 'success': False,
        }

    finally:
        # ✅ 多轮清理确保内存释放
        if 'model' in dir() and model is not None:
            del model
        if 'trainer' in dir() and trainer is not None:
            del trainer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        # 二次清理，确保彻底
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def calculate_multi_objective_score(
        acc, ari, nmi, topk_max_silhouette, topk_min_dbi,
        weights=None
):
    """
    计算多目标综合评分

    参数：
        acc                 - 聚类准确率 [0, 1]，越大越好
        ari                 - 调整兰德指数 [-1, 1]，越大越好
        nmi                 - 标准化互信息 [0, 1]，越大越好
        topk_max_silhouette - 轮廓系数 [-1, 1]，越大越好
        topk_min_dbi       - Davies-Bouldin指数 [0, +∞)，越小越好
        weights            - 权重字典，默认为均匀权重

    返回：
        综合评分 [0, 1]，越大越好
    """
    if weights is None:
        # 默认权重：外在指标(ACC/ARI/NMI)占70%，内在指标(Sil/DBI)占30%
        weights = {
            'acc': 0.25,  # 外在主指标
            'ari': 0.25,  # 外在辅助指标
            'nmi':  0.20,  # 外在辅助指标
            'silhouette': 0.15,  # 内在几何结构
            'dbi': 0.15,  # 内在紧密度
        }

        # ── 指标标准化处理 ──
    # ACC: [0, 1] → [0, 1] (无需变换)
    acc_norm = max(0.0, min(1.0, acc))

    # ARI: [-1, 1] → [0, 1]
    ari_norm = max(0.0, min(1.0, (ari + 1.0) / 2.0))

    # NMI: [0, 1] → [0, 1] (无需变换)
    nmi_norm = max(0.0, min(1.0, nmi))

    # Silhouette: [-1, 1] → [0, 1]
    sil_norm = max(0.0, min(1.0, (topk_max_silhouette + 1.0) / 2.0))

    # DBI: [0, +∞) → [0, 1]，使用倒数变换
    # 优秀DBI < 1，良好DBI < 2，较差DBI > 3
    dbi_norm = max(0.0, min(1.0, 1.0 / (1.0 + topk_min_dbi)))

    # ── 加权综合评分 ──
    composite_score = (
            weights['acc'] * acc_norm +
            weights['ari'] * ari_norm +
            weights['nmi'] * nmi_norm +
            weights['silhouette'] * sil_norm +
            weights['dbi'] * dbi_norm
    )

    return float(composite_score)

# ================================================================
# ✅ IDC+SEFS 批量贝叶斯优化器
#
# 与 IDC 原版优化器的主要区别：
#   1. 搜索空间新增 sefs_tau（SEFS 温度参数，核心超参数）
#   2. 移除与 IDC+SEFS 无关的 ComFS 参数（beta_s/beta_d 等）
#   3. 固定 use_sefs=True，专门针对 SEFS 模型优化
#   4. local_gates_lambda 搜索范围上调（SEFS 需要更强的稀疏正则化）
# ================================================================
class SEFSBayesianOptimizer:
    """
    IDC+SEFS 批量贝叶斯超参数优化器

    工作流程：
    1. 定义搜索空间（18 个超参数）
    2. 高斯过程（GP）模型根据历史结果预测参数性能
    3. EI（期望改进）采集函数选择下一批候选参数
    4. 并行评估候选参数（多进程 + 多种子）
    5. 将多种子平均 ACC 反馈给 GP 模型
    6. 重复直到达到总采样次数
    """

    # ── 搜索空间定义（以 ALLAML.yaml 精调值为中心的局部搜索）──
    # Round 1 后微调：4 个参数碰到边界，已扩展；其余不变
    #
    # 参数名               范围                    Round1最佳    说明
    # ───────────────────────────────────────────────────────────
    # batch_size           [4, 14]                10(碰上限)    扩至14
    # correlation_threshold [0.04, 0.12]          0.0766
    # epochs               [900, 1500]            1052(碰下限)  降至900
    # ae_non_gated_epochs  [40, 75]               70
    # ae_pretrain_epochs   [450, 700]             627
    # start_global_*       [700, 900]             750
    # gates_hidden_dim     [80, 180]              151
    # lr_pretrain          [5e-4, 1.2e-3]         6.36e-4
    # lr_clustering        [2e-3, 6e-3]           3.75e-3
    # lr_aux_classifier    [5e-5, 2e-4]           9.92e-5
    # sched_pretrain_min_lr [1e-5, 4e-5]          2.09e-5
    # sched_clustering_min_lr [2e-7, 1e-6]        3.13e-7
    # local_gates_lambda   [0.7, 2.0]             0.925
    # global_gates_lambda  [5e-5, 5e-4]           1.25e-4
    # gtcr_lambda          [0.015, 0.06]          0.0402
    # tau                  [35, 90]               69
    # eps                  [0.2, 0.6]             0.487(近上限) 扩至0.6
    # mask_percentage      [0.45, 0.8]            0.719
    # sefs_tau             [0.5, 1.8]             0.871(近下限) 降至0.5
    #   sefs_tau: 小→门控更硬（稀疏），大→门控更软（平滑）

    ALLAML_SEARCH_SPACE = [
        # ── 以 cfg/ALLAML.yaml 精调值为中心的局部搜索 ──
        # Round 1 后微调：4 个参数碰到边界，已扩展；其余不变
        Integer(4, 14, name='batch_size'),                            # Round1最佳=10(碰上限)→扩至14
        Real(0.04, 0.12, name='correlation_threshold'),              # 当前 0.075
        Integer(900, 1500, name='epochs'),                            # Round1最佳=1052(碰下限)→降至900
        Integer(40, 75, name='ae_non_gated_epochs'),                 # 当前 58
        Integer(450, 700, name='ae_pretrain_epochs'),                # 当前 594
        Integer(700, 900, name='start_global_gates_training_on_epoch'), # 当前 812
        Integer(80, 180, name='gates_hidden_dim'),                   # 当前 120
        Real(5e-4, 1.2e-3, 'log-uniform', name='lr_pretrain'),      # 当前 8e-4
        Real(2e-3, 6e-3, 'log-uniform', name='lr_clustering'),      # 当前 4e-3
        Real(5e-5, 2e-4, 'log-uniform', name='lr_aux_classifier'),  # 当前 1e-4
        Real(1e-5, 4e-5, 'log-uniform', name='sched_pretrain_min_lr'), # 当前 2e-5
        Real(2e-7, 1e-6, 'log-uniform', name='sched_clustering_min_lr'), # 当前 5e-7
        Real(0.7, 2.0, name='local_gates_lambda'),                   # 当前 1.21
        Real(5e-5, 5e-4, name='global_gates_lambda'),               # 当前 2e-4
        Real(0.015, 0.06, name='gtcr_lambda'),                       # 当前 0.033
        Real(35, 90, name='tau'),                                     # 当前 60
        Real(0.2, 0.6, name='eps'),                                   # Round1最佳=0.487(近上限)→扩至0.6
        Real(0.45, 0.8, name='mask_percentage'),                     # 当前 0.63
        Real(0.5, 1.8, name='sefs_tau'),                             # Round1最佳=0.871(近下限)→降至0.5
    ]

    SRBCT_SEARCH_SPACE = [
        # ── SRBCT 精调：围绕第一轮最佳（第10轮 ACC=0.5554）缩小范围 ──
        # 第一轮最佳值 → 精调范围（碰边界的参数向边界外扩展）
        Integer(4, 8, name='batch_size'),                            # 最佳 5
        Real(0.06, 0.12, name='correlation_threshold'),              # 最佳 0.088
        Integer(480, 750, name='epochs'),                            # 最佳 587
        Integer(35, 55, name='ae_non_gated_epochs'),                 # 最佳 43
        Integer(170, 280, name='ae_pretrain_epochs'),                # 最佳 221
        Integer(450, 620, name='start_global_gates_training_on_epoch'), # 最佳 556(近上界)
        Integer(90, 150, name='gates_hidden_dim'),                   # 最佳 118
        Real(3.2e-4, 7e-4, 'log-uniform', name='lr_pretrain'),      # 最佳 4.24e-4(近下界)
        Real(2.5e-3, 5e-3, 'log-uniform', name='lr_clustering'),    # 最佳 4.0e-3
        Real(6e-5, 1.5e-4, 'log-uniform', name='lr_aux_classifier'), # 最佳 9.4e-5
        Real(1.5e-5, 4e-5, 'log-uniform', name='sched_pretrain_min_lr'), # 最佳 2.5e-5
        Real(2e-7, 1e-6, 'log-uniform', name='sched_clustering_min_lr'), # 最佳 3.2e-7(近下界)
        Real(0.35, 0.9, name='local_gates_lambda'),                  # 最佳 0.543(近下界)
        Real(1e-4, 3e-4, name='global_gates_lambda'),               # 最佳 2.0e-4
        Real(0.04, 0.12, name='gtcr_lambda'),                        # 最佳 0.078(近上界)
        Real(35, 70, name='tau'),                                     # 最佳 50.6
        Real(0.08, 0.2, name='eps'),                                  # 最佳 0.126
        Real(0.45, 0.65, name='mask_percentage'),                    # 最佳 0.567
        Real(1.0, 1.6, name='sefs_tau'),                             # 最佳 1.338
    ]

    PROSTATE_SEARCH_SPACE = [
        # ── Prostate 精调：围绕第一轮最佳（第14轮 ACC=0.6725）缩小范围 ──
        # 碰边界的参数向边界外扩展
        Integer(6, 11, name='batch_size'),                            # 最佳 8
        Real(0.05, 0.11, name='correlation_threshold'),              # 最佳 0.0767
        Integer(900, 1300, name='epochs'),                            # 最佳 1085(近上界)
        Integer(45, 70, name='ae_non_gated_epochs'),                 # 最佳 60(近上界)
        Integer(120, 220, name='ae_pretrain_epochs'),                # 最佳 161
        Integer(600, 800, name='start_global_gates_training_on_epoch'), # 最佳 703
        Integer(120, 170, name='gates_hidden_dim'),                   # 最佳 151
        Real(4e-4, 9e-4, 'log-uniform', name='lr_pretrain'),        # 最佳 6.02e-4
        Real(1.5e-3, 4e-3, 'log-uniform', name='lr_clustering'),    # 最佳 2.52e-3
        Real(7e-5, 1.6e-4, 'log-uniform', name='lr_aux_classifier'), # 最佳 1.155e-4
        Real(2e-5, 6e-5, 'log-uniform', name='sched_pretrain_min_lr'), # 最佳 3.83e-5
        Real(5e-7, 2e-6, 'log-uniform', name='sched_clustering_min_lr'), # 最佳 9.66e-7(近上界)
        Real(0.5, 1.2, name='local_gates_lambda'),                   # 最佳 0.715(近下界)
        Real(8e-5, 3e-4, name='global_gates_lambda'),               # 最佳 1.62e-4
        Real(0.06, 0.13, name='gtcr_lambda'),                        # 最佳 0.0949
        Real(35, 60, name='tau'),                                     # 最佳 46.1
        Real(0.15, 0.4, name='eps'),                                  # 最佳 0.275(近上界)
        Real(0.4, 0.65, name='mask_percentage'),                     # 最佳 0.549
        Real(0.6, 1.3, name='sefs_tau'),                             # 最佳 0.807(近下界)
    ]

    # ✅ 按 dataset 名选择搜索空间
    SEARCH_SPACES = {
        'ALLAML': ALLAML_SEARCH_SPACE,
        'SRBCT': SRBCT_SEARCH_SPACE,
        'PROSTATE': PROSTATE_SEARCH_SPACE,
    }

    PARAM_NAMES = ['batch_size', 'correlation_threshold',
        'epochs', 'ae_non_gated_epochs', 'ae_pretrain_epochs',
        'start_global_gates_training_on_epoch',
        'gates_hidden_dim',
        'lr_pretrain', 'lr_clustering', 'lr_aux_classifier',
        'sched_pretrain_min_lr', 'sched_clustering_min_lr',
        'local_gates_lambda', 'global_gates_lambda', 'gtcr_lambda',
        'tau', 'eps', 'mask_percentage',
        'sefs_tau',   # ← IDC+SEFS 专有
    ]

    def __init__(
            self,
            base_cfg_path: str,
            n_calls:       int  = 40,
            batch_size:    int  = 4,
            n_workers:     int  = 4,
            n_seeds:       int  = 3,
            random_state:  int  = 42,
            early_stop_config: dict = None,
            resume_dir:    str  = None,  # ✅ 断点续跑：指定之前的输出目录
            stability_lambda: float = 0.3,  # ✅ 方向A：方差惩罚系数 λ（0=只看均值）
    ):
        """
        初始化优化器

        参数：
            base_cfg_path    - yaml 配置文件路径（如 cfg/test.yaml）
            n_calls          - 总采样次数（建议 ≥ 40）
            batch_size       - 每批并行评估的参数组数（建议 = n_workers）
            n_workers        - 并行子进程数（不超过 GPU 数量 × 2）
            n_seeds          - 每组参数重复的随机种子数（建议 3~5）
            random_state     - GP 模型的随机种子（保证可重复）
            early_stop_config - 早停配置字典
            stability_lambda - 方向A：稳定性惩罚系数。优化目标 = 均值 − λ·标准差。
                               λ=0 退化为只看均值；λ 越大越偏好低方差。
        """
        self.base_cfg      = OmegaConf.load(base_cfg_path)
        self.cfg_dict      = OmegaConf.to_container(self.base_cfg, resolve=True)

        # ✅ 按 dataset 名选择搜索空间（ALLAML / SRBCT 各自一套）
        dataset_name = str(self.base_cfg.get('dataset', 'ALLAML')).upper()
        if dataset_name in self.SEARCH_SPACES:
            self.search_space = self.SEARCH_SPACES[dataset_name]
            print(f"🎯 使用搜索空间: {dataset_name}（{len(self.search_space)} 个参数）")
        else:
            self.search_space = self.ALLAML_SEARCH_SPACE
            print(f"⚠️ 未找到 '{dataset_name}' 的搜索空间，回退到 ALLAML")

        self.n_calls       = n_calls
        self.batch_size    = batch_size
        self.n_workers     = min(n_workers, mp.cpu_count())
        self.n_seeds       = n_seeds
        self.random_state  = random_state
        self.stability_lambda = stability_lambda  # ✅ 方向A：方差惩罚系数
        self.n_iterations  = math.ceil(n_calls / batch_size)

        self.early_stop_config = early_stop_config or {'enabled': False}
        self.results_log   = []   # 存储所有评估结果
        self.per_seed_log  = []   # ✅ 每个种子的独立结果
        self._current_iteration = 0  # ✅ 当前轮次追踪
        self._start_iteration = 1     # ✅ 起始轮次（续跑时 > 1）
        self._completed_iteration = 0  # ✅ 最近一个完整完成的轮次（tell 完成）
        self._resuming = resume_dir is not None

        # ── 输出目录 ──
        if resume_dir and os.path.isdir(resume_dir):
            self.output_dir = resume_dir
            print(f"📂 断点续跑模式，结果目录: {self.output_dir}/")
        else:
            timestamp        = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir  = f"sefs_bayes_opt_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"📁 结果将保存至: {self.output_dir}/")

        # ── 创建高斯过程优化器 ──
        self.optimizer = Optimizer(
            dimensions=self.search_space,
            base_estimator='GP',
            acq_func='EI',
            acq_optimizer='sampling',
            random_state=self.random_state,
        )

        # ── 断点续跑：加载历史结果并恢复 GP 状态 ──
        if self._resuming:
            self._restore_state()
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        # ── 即时保存文件路径 ──
        self._per_seed_csv_path = os.path.join(self.output_dir, 'per_seed_immediate.csv')
        self._per_seed_lock = threading.Lock()
        if not self._resuming or not os.path.exists(self._per_seed_csv_path):
            with open(self._per_seed_csv_path, 'w', encoding='utf-8') as f:
                f.write('iteration,param_group,seed,success,acc,ari,nmi,'
                        'multi_objective_score,actual_epochs,early_stopped,'
                        'topk_max_silhouette,topk_min_dbi,'
                        + ','.join(self.PARAM_NAMES) + '\n')
                f.flush()
                os.fsync(f.fileno())

        self._print_init_info()

    def _restore_state(self):
        """✅ 断点续跑：从磁盘恢复 GP 优化器状态和结果日志"""
        checkpoint_path = os.path.join(self.output_dir, 'optimizer_checkpoint.json')

        if not os.path.exists(checkpoint_path):
            print("⚠️ 未找到 optimizer_checkpoint.json，从头开始优化")
            self._resuming = False
            os.makedirs(self.output_dir, exist_ok=True)
            return

        # 1) 加载 checkpoint
        with open(checkpoint_path, 'r') as f:
            ckpt = json.load(f)
        Xi_loaded = ckpt.get('Xi', [])
        yi_loaded = ckpt.get('yi', [])
        completed_iteration = ckpt.get('completed_iteration', 0)

        print(f"📂 加载 checkpoint: {len(Xi_loaded)} 组已评估参数, "
              f"已完成 {completed_iteration} 轮")

        # 2) 恢复 GP 状态：逐个 tell 历史数据
        for xi, yi in zip(Xi_loaded, yi_loaded):
            self.optimizer.tell(xi, yi)

        # 3) 恢复结果日志（从 checkpoint 中直接恢复，而非依赖 all_results.csv）
        results_loaded = ckpt.get('results_log', [])
        if results_loaded:
            for r in results_loaded:
                self.results_log.append({
                    'iteration': int(r['iteration']),
                    'params': r['params'],
                    'acc': r['acc'], 'ari': r['ari'], 'nmi': r['nmi'],
                    'topk_max_silhouette': r.get('topk_max_silhouette', 0),
                    'topk_min_dbi': r.get('topk_min_dbi', 5),
                    'multi_objective_score': r.get('multi_objective_score', 0),
                    'multi_objective_std': r.get('multi_objective_std', 0),
                    'stability_score': r.get('stability_score', r.get('multi_objective_score', 0)),
                    'actual_epochs': r.get('actual_epochs', 0),
                    'early_stopped': r.get('early_stopped', False),
                    'n_successful': r.get('n_successful', 0),
                    'n_total': r.get('n_total', 0),
                })
            print(f"📂 加载 {len(self.results_log)} 条历史聚合结果")

        # 4) 恢复 per_seed_log（仅保留已完成轮次，丢弃被中断轮次的残片）
        per_seed_path = os.path.join(self.output_dir, 'per_seed_immediate.csv')
        if os.path.exists(per_seed_path):
            import pandas as pd
            df = pd.read_csv(per_seed_path)
            # 只保留已完成轮次的记录，被 OOM 打断的轮次（iteration > completed_iteration）将被重新评估
            df = df[df['iteration'] <= completed_iteration]
            self.per_seed_log = df.to_dict('records')
            print(f"📂 加载 {len(self.per_seed_log)} 条 per-seed 记录"
                  f"（丢弃被中断轮次的残片）")

        # 5) 设置续跑起点
        self._completed_iteration = completed_iteration
        self._start_iteration = completed_iteration + 1
        remaining = self.n_iterations - completed_iteration
        print(f"✅ 续跑: 从第 {self._start_iteration} 轮开始, "
              f"剩余 {remaining} 轮 ({self.batch_size * remaining} 组参数)\n")

    def _save_checkpoint(self, completed_iteration: int):
        """✅ 保存 GP 优化器状态，支持断点续跑（可在每个 seed 后调用）"""
        checkpoint_path = os.path.join(self.output_dir, 'optimizer_checkpoint.json')
        # results_log 中 params 可能含 numpy 类型，需转成可 JSON 序列化的 float
        results_serializable = []
        for r in self.results_log:
            results_serializable.append({
                'iteration': r['iteration'],
                'params': [float(v) for v in r['params']],
                'acc': r['acc'], 'ari': r['ari'], 'nmi': r['nmi'],
                'topk_max_silhouette': r.get('topk_max_silhouette', 0),
                'topk_min_dbi': r.get('topk_min_dbi', 5),
                'multi_objective_score': r.get('multi_objective_score', 0),
                'multi_objective_std': r.get('multi_objective_std', 0),
                'stability_score': r.get('stability_score', r.get('multi_objective_score', 0)),
                'actual_epochs': r.get('actual_epochs', 0),
                'early_stopped': r.get('early_stopped', False),
                'n_successful': r.get('n_successful', 0),
                'n_total': r.get('n_total', 0),
            })
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'completed_iteration': completed_iteration,
                'n_evaluated': len(self.optimizer.Xi),
                'Xi': [[float(v) for v in xi] for xi in self.optimizer.Xi],
                'yi': [float(y) for y in self.optimizer.yi],
                'param_names': self.PARAM_NAMES,
                'results_log': results_serializable,
                'timestamp': datetime.now().isoformat(),
            }, f)
        # 不打印日志——轮次保存已有提示

    def _save_result_immediately(self, iteration, param_idx, seed, result, params):
        """✅ 每个 seed 完成后立即写入磁盘 + 内存 log，防止 OOM 丢失"""
        row = {
            'iteration': iteration,
            'param_group': param_idx,
            'seed': seed,
            'success': result['success'],
            'acc': result['acc'],
            'ari': result['ari'],
            'nmi': result['nmi'],
            'multi_objective_score': result.get('multi_objective_score', 0.0),
            'actual_epochs': result.get('actual_epochs', 0),
            'early_stopped': result.get('early_stopped', False),
            'topk_max_silhouette': result.get('topk_max_silhouette', float('nan')),
            'topk_min_dbi': result.get('topk_min_dbi', float('nan')),
        }
        row.update(dict(zip(self.PARAM_NAMES, params)))

        # 1) 内存 log
        self.per_seed_log.append(row)

        # 2) 立即写入磁盘（线程安全）
        with self._per_seed_lock:
            with open(self._per_seed_csv_path, 'a', encoding='utf-8') as f:
                vals = [str(row[k]) for k in [
                    'iteration', 'param_group', 'seed', 'success', 'acc', 'ari',
                    'nmi', 'multi_objective_score', 'actual_epochs', 'early_stopped',
                    'topk_max_silhouette', 'topk_min_dbi'
                ] + self.PARAM_NAMES]
                f.write(','.join(vals) + '\n')
                f.flush()
                os.fsync(f.fileno())  # 强制刷到磁盘

        # 3) ✅ 每个 seed 完成后立即保存 checkpoint（GP 状态 + 已完成的聚合结果）
        #    即使 OOM 打断在轮次中间，已完成的轮次也不会丢失
        self._save_checkpoint(self._completed_iteration)

    def _print_init_info(self):
        print("\n" + "=" * 60)
        print("🚀 IDC+SEFS 贝叶斯超参数优化器")
        print("=" * 60)
        print(f"   搜索参数数量  : {len(self.PARAM_NAMES)}")
        print(f"   总采样次数    : {self.n_calls}")
        print(f"   每批参数组数  : {self.batch_size}")
        print(f"   并行进程数    : {self.n_workers}")
        print(f"   每组种子数    : {self.n_seeds}")
        print(f"   总评估次数    : {self.n_calls * self.n_seeds}")
        if self.early_stop_config.get('enabled'):
            print(f"   早停监控指标  : {self.early_stop_config['monitor']}")
            print(f"   早停容忍轮数  : {self.early_stop_config['patience']}")
        print("=" * 60 + "\n")

    # ────────────────────────────────────────────────────────
    # ✅ 并行评估：对一批参数组并行运行多个种子
    # ────────────────────────────────────────────────────────
    def _evaluate_batch_parallel(self, params_list):
        """
        并行评估一批候选参数

        实现细节：
        - 每组参数 × n_seeds 个种子 = 总任务数
        - ProcessPoolExecutor 并行执行，避免 GIL 限制
        - 每个子进程独立使用 GPU（通过不同 seed 区分）
        - 结果按参数组聚合（取多种子平均值）

        返回：每组参数的聚合结果列表
        """
        # ── 构建任务列表：(参数组idx, 参数值, 种子) ──
        tasks = [
            {'param_idx': pi, 'params': params, 'seed': s}
            for pi, params in enumerate(params_list)
            for s in range(self.n_seeds)
        ]
        print(f"   📦 提交 {len(tasks)} 个并行任务 "
              f"({len(params_list)} 组参数 × {self.n_seeds} 种子)")

        # ── 并行执行 ──
        raw_results = {}  # {param_idx: [result1, result2, ...]}

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_task = {
                executor.submit(
                    evaluate_single_params,
                    self.cfg_dict,
                    *task['params'],    # 解包参数列表（顺序需与 PARAM_NAMES 一致）
                    task['seed'],
                    self.early_stop_config
                ): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                pi   = task['param_idx']
                try:
                    result = future.result()
                    if pi not in raw_results:
                        raw_results[pi] = []
                    result['_seed'] = task['seed']
                    raw_results[pi].append(result)

                    # ✅ 每个 seed 完成后立即保存到磁盘！
                    self._save_result_immediately(
                        self._current_iteration, pi, task['seed'],
                        result, task['params']
                    )

                    status = "✅" if result['success'] else "❌"
                    multi_score = result.get('multi_objective_score', 0.0)
                    print(f"   {status} 参数组{pi} Seed{task['seed']}: "
                          f"综合评分={multi_score:.4f} "
                          f"(ACC={result['acc']:.3f}, ARI={result['ari']:.3f}, "
                          f"NMI={result['nmi']:.3f})")
                except Exception as e:
                    print(f"   ❌ 参数组{pi} Seed{task['seed']} 异常: {e}")
                    if pi not in raw_results:
                        raw_results[pi] = []
                    fail_result = {
                        'acc': 0., 'ari': 0., 'nmi': 0.,
                        'topk_max_silhouette': 0., 'topk_min_dbi': 5.0,
                        'multi_objective_score': 0.0,
                        'actual_epochs': 0, 'early_stopped': False, 'success': False,
                        '_seed': task['seed']
                    }
                    raw_results[pi].append(fail_result)
                    # ✅ 失败结果也立即保存
                    self._save_result_immediately(
                        self._current_iteration, pi, task['seed'],
                        fail_result, task['params']
                    )

                    # ── 聚合：对每组参数取成功种子的均值 ──
        aggregated = []
        for pi in range(len(params_list)):
            group = raw_results.get(pi, [])
            success = [r for r in group if r['success']]

            if success:
                scores = [r['multi_objective_score'] for r in success]
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores))
                # ✅ 方向A：稳定性调整后的目标 = 均值 − λ·标准差
                #   λ=0 → 只看均值（原行为）；λ 越大 → 越惩罚高方差，偏好稳定解
                stability_score = mean_score - self.stability_lambda * std_score
                agg = {
                    'success': True,
                    'acc': np.mean([r['acc'] for r in success]),
                    'ari': np.mean([r['ari'] for r in success]),
                    'nmi': np.mean([r['nmi'] for r in success]),
                    'topk_max_silhouette': np.mean([r['topk_max_silhouette'] for r in success]),
                    'topk_min_dbi': np.mean([r['topk_min_dbi'] for r in success]),
                    'multi_objective_score': mean_score,
                    'multi_objective_std': std_score,
                    'stability_score': stability_score,  # ✅ 方向A：GP 优化目标
                    'actual_epochs': np.mean([r['actual_epochs'] for r in success]),
                    'early_stopped': any(r['early_stopped'] for r in success),
                    'n_successful': len(success),
                    'n_total': self.n_seeds,
                }
                print(f"   📊 参数组{pi} 聚合: 稳定评分={stability_score:.4f} "
                      f"(综合={mean_score:.4f}±{std_score:.4f}, "
                      f"ACC={agg['acc']:.3f}±{np.std([r['acc'] for r in success]):.3f})")
            else:
                agg = {
                    'success': False,
                    'acc': 0., 'ari': 0., 'nmi': 0.,
                    'topk_max_silhouette': 0., 'topk_min_dbi': 5.0,
                    'multi_objective_score': 0.0,
                    'actual_epochs': 0, 'early_stopped': False,
                    'n_successful': 0, 'n_total': self.n_seeds,
                }
                print(f"   ❌ 参数组{pi}: 所有种子均失败")

            aggregated.append(agg)

        return aggregated

    # ────────────────────────────────────────────────────────
    # ✅ 主优化循环
    # ────────────────────────────────────────────────────────
    def optimize(self):
        """
        执行贝叶斯优化主循环

        每轮迭代：
        1. optimizer.ask()  → GP 推荐 batch_size 组候选参数
        2. _evaluate_batch_parallel() → 并行评估（含多种子）
        3. optimizer.tell() → 将结果反馈给 GP 模型
        4. 保存中间结果（防止意外中断丢失数据）
        """
        print(f"🔍 开始优化（共 {self.n_iterations} 轮，"
              f"从第 {self._start_iteration} 轮开始）\n")

        for iteration in range(self._start_iteration, self.n_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"📍 第 {iteration}/{self.n_iterations} 轮")
            print(f"{'=' * 60}")

            # ── Step 1: GP 推荐候选参数 ──
            candidates = self.optimizer.ask(n_points=self.batch_size)
            print(f"   GP 推荐 {len(candidates)} 组候选参数")

            # ── Step 2: 并行评估 ──
            self._current_iteration = iteration  # ✅ 新增：让 per_seed_log 能记录轮次
            batch_results = self._evaluate_batch_parallel(candidates)

            # ── Step 3: 将结果告知 GP（最小化负稳定评分 = 最大化稳定评分）──
            for pi, (params, result) in enumerate(zip(candidates, batch_results)):
                if result['success']:
                    # ✅ 方向A：用稳定性调整后的目标（均值 − λ·std）作为 GP 优化目标
                    loss = -result['stability_score']
                    self.optimizer.tell(params, loss)

                    # 记录完整结果（含稳定性指标）
                    self.results_log.append({
                        'iteration': iteration,
                        'params': params,
                        'acc': result['acc'],
                        'ari': result['ari'],
                        'nmi': result['nmi'],
                        'topk_max_silhouette': result['topk_max_silhouette'],
                        'topk_min_dbi': result['topk_min_dbi'],
                        'multi_objective_score': result['multi_objective_score'],
                        'multi_objective_std': result['multi_objective_std'],
                        'stability_score': result['stability_score'],
                        'actual_epochs': result['actual_epochs'],
                        'early_stopped': result['early_stopped'],
                        'n_successful': result['n_successful'],
                        'n_total': result['n_total'],
                    })


                    # ── Step 4: 打印当前全局最佳 ──
            if self.results_log:
                # 根据稳定评分排序（方向A：偏好均值高且方差低的解）
                best = max(self.results_log, key=lambda x: x['stability_score'])
                print(f"\n🏆 全局最佳 (第{best['iteration']}轮发现):")
                print(f"   稳定评分: {best['stability_score']:.4f} "
                      f"(综合={best['multi_objective_score']:.4f}±{best['multi_objective_std']:.4f})")
                print(f"   ACC: {best['acc']:.4f}, ARI: {best['ari']:.4f}, NMI: {best['nmi']:.4f}")
                print(f"   Silhouette: {best['topk_max_silhouette']:.4f}, DBI: {best['topk_min_dbi']:.4f}")

                best_params_dict = dict(zip(self.PARAM_NAMES, best['params']))
                print(f"   关键参数: sefs_tau={best_params_dict['sefs_tau']:.3f}, "
                      f"local_gates_lambda={best_params_dict['local_gates_lambda']:.3f}")

                # ── Step 5: 每轮保存中间结果 + GP 状态 checkpoint ──
            self._save_intermediate(iteration)
            self._completed_iteration = iteration  # ✅ 本轮 tell 完成，标记为已完成
            self._save_checkpoint(iteration)  # ✅ 断点续跑 checkpoint

        # ── 最终保存完整报告 ──
        self._save_final_report()
        return self._get_best_result_multi_objective()

    # ────────────────────────────────────────────────────────
    # ✅ 中间结果保存（每轮迭代后调用）
    # ────────────────────────────────────────────────────────
    def _save_intermediate(self, iteration: int):
        """每轮优化结束后立即保存，防止中断丢失数据"""
        if not self.results_log:
            return

        best = max(self.results_log, key=lambda x: x['stability_score'])

        # ── 写入 TXT（人类可读）──
        txt_path = os.path.join(self.output_dir, f"iter_{iteration:03d}_best.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"第 {iteration} 轮优化中间结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"稳定评分          : {best['stability_score']:.6f}\n")
            f.write(f"综合评分          : {best['multi_objective_score']:.6f} ± {best['multi_objective_std']:.6f}\n")
            f.write(f"─────────────────────────────────────\n")
            f.write(f"聚类准确率 (ACC)  : {best['acc']:.6f}\n")
            f.write(f"调整兰德指数(ARI) : {best['ari']:.6f}\n")
            f.write(f"标准化互信息(NMI) : {best['nmi']:.6f}\n")
            f.write(f"轮廓系数(Silhouette): {best['topk_max_silhouette']:.6f}\n")
            f.write(f"DBI指数(Davies-Bouldin): {best['topk_min_dbi']:.6f}\n")
            f.write(f"发现于第          : {best['iteration']} 轮\n\n")
            f.write("最佳参数:\n")
            for name, val in zip(self.PARAM_NAMES, best['params']):
                f.write(f"  {name:45s}: {val}\n")

        print(f"   💾 多目标中间结果已保存: {txt_path}")
        # ✅ 新增：保存本轮及历史所有 per-seed 明细 CSV
        if self.per_seed_log:
            per_seed_path = os.path.join(self.output_dir, 'per_seed_all_results.csv')
            pd.DataFrame(self.per_seed_log).to_csv(per_seed_path, index=False)
            print(f"   📋 Per-seed 明细已保存: {per_seed_path}")


            # # ── 写入 JSON（编程读取）──
        # json_path = os.path.join(self.output_dir, f"iter_{iteration:03d}_best.json")
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         'iteration':    iteration,
        #         'timestamp':    datetime.now().isoformat(),
        #         'best_acc':     float(best['acc']),
        #         'best_ari':     float(best['ari']),
        #         'best_nmi':     float(best['nmi']),
        #         'found_at_iter': int(best['iteration']),
        #         'params':       dict(zip(self.PARAM_NAMES,
        #                                  [float(v) for v in best['params']])),
        #     }, f, indent=4)
        #
        # print(f"   💾 中间结果已保存: {txt_path}")

    # ────────────────────────────────────────────────────────
    # ✅ 最终完整报告
    # ────────────────────────────────────────────────────────
    def _save_final_report(self):
        """优化结束后保存完整报告（CSV + JSON + TXT）"""
        if not self.results_log:
            return

        # ── 1. CSV 汇总（可用 Excel 打开）──
        rows = []
        for r in self.results_log:
            row = {
                'iteration': r['iteration'],
                'multi_objective_score': r['multi_objective_score'],
                'acc': r['acc'],
                'ari': r['ari'],
                'nmi': r['nmi'],
                'topk_max_silhouette': r['topk_max_silhouette'],
                'topk_min_dbi': r['topk_min_dbi'],
                'actual_epochs': r['actual_epochs'],
                'early_stopped': r['early_stopped'],
                'n_successful': r['n_successful'],
                'n_total': r['n_total'],
            }
            row.update(dict(zip(self.PARAM_NAMES, r['params'])))
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('acc', ascending=False)
        csv_path = os.path.join(self.output_dir, 'all_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n📊 完整结果已保存: {csv_path}")
        # ✅ 新增：保存完整 per-seed 明细 CSV（最终版）
        if self.per_seed_log:
            per_seed_df = pd.DataFrame(self.per_seed_log)
            # 按 iteration → param_group → seed 排序，方便阅读
            per_seed_df = per_seed_df.sort_values(
                ['iteration', 'param_group', 'seed']
            ).reset_index(drop=True)
            per_seed_final_path = os.path.join(self.output_dir, 'per_seed_all_results_final.csv')
            per_seed_df.to_csv(per_seed_final_path, index=False)
            print(f"📋 Per-seed 完整明细已保存: {per_seed_final_path}")

            # ✅ 同时保存一份人类可读的 per-seed 摘要 TXT
            per_seed_txt_path = os.path.join(self.output_dir, 'per_seed_summary.txt')
            with open(per_seed_txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("Per-Seed 明细记录（每个种子的独立评估结果）\n")
                f.write("=" * 70 + "\n\n")
                for iteration_num in per_seed_df['iteration'].unique():
                    f.write(f"── 第 {iteration_num} 轮 ──\n")
                    iter_df = per_seed_df[per_seed_df['iteration'] == iteration_num]
                    for _, row in iter_df.iterrows():
                        status = "✅" if row['success'] else "❌"
                        f.write(
                            f"  {status} 参数组{int(row['param_group']):02d} "
                            f"Seed{int(row['seed'])}: "
                            f"ACC={row['acc']:.4f}, "
                            f"ARI={row['ari']:.4f}, "
                            f"NMI={row['nmi']:.4f}, "
                            f"Sil={row['topk_max_silhouette']:.4f}, "
                            f"DBI={row['topk_min_dbi']:.4f}"
                        )
                        if row['early_stopped']:
                            f.write(f" [早停@epoch{int(row['actual_epochs'])}]")
                        f.write("\n")
                        # 打印该轮的聚合对比
                    f.write(f"\n  {'参数组':>5} {'成功种子':>8} {'ACC均值':>10} {'ACC标准差':>10}\n")
                    for pg in iter_df['param_group'].unique():
                        pg_df = iter_df[(iter_df['param_group'] == pg) & (iter_df['success'] == True)]
                        if len(pg_df) > 0:
                            f.write(
                                f"  {int(pg):>5} {len(pg_df):>8} "
                                f"{pg_df['acc'].mean():>10.4f} "
                                f"{pg_df['acc'].std():>10.4f}\n"
                            )
                    f.write("\n")
            print(f"📝 Per-seed 摘要报告已保存: {per_seed_txt_path}")
        # ── 2. 最佳参数 JSON ──
        best = max(self.results_log, key=lambda x: x['stability_score'])
        best_json_path = os.path.join(self.output_dir, 'multi_objective_best_params.json')
        with open(best_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'stability_score': float(best['stability_score']),
                'multi_objective_score': float(best['multi_objective_score']),
                'multi_objective_std': float(best['multi_objective_std']),
                'metrics': {
                    'acc': float(best['acc']),
                    'ari': float(best['ari']),
                    'nmi': float(best['nmi']),
                    'topk_max_silhouette': float(best['topk_max_silhouette']),
                    'topk_min_dbi': float(best['topk_min_dbi']),
                },
                'found_at_iteration': int(best['iteration']),
                'params': dict(zip(self.PARAM_NAMES,
                                   [float(v) for v in best['params']])),
            }, f, indent=4, ensure_ascii=False)
        print(f"🏆 多目标最佳参数已保存: {best_json_path}")

        # ── 3. 人类可读报告 TXT ──
        report_path = os.path.join(self.output_dir, 'multi_objective_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("IDC+SEFS 贝叶斯优化完整报告（方向A：稳定性目标）\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总采样次数     : {self.n_calls}\n")
            f.write(f"优化目标       : 稳定评分 = 综合评分均值 − λ·标准差 (λ={self.stability_lambda})\n")
            f.write(f"成功评估参数组 : {len(self.results_log)}\n\n")

            f.write("=" * 70 + "\n")
            f.write("最佳结果 (按稳定评分排序)\n")
            f.write("=" * 70 + "\n")
            f.write(f"稳定评分       : {best['stability_score']:.6f}\n")
            f.write(f"综合评分       : {best['multi_objective_score']:.6f} ± {best['multi_objective_std']:.6f}\n")
            f.write(f"─────────────────────────────────────\n")
            f.write(f"ACC (准确率)   : {best['acc']:.6f}\n")
            f.write(f"ARI (调整兰德) : {best['ari']:.6f}\n")
            f.write(f"NMI (标准互信息): {best['nmi']:.6f}\n")
            f.write(f"Silhouette     : {best['topk_max_silhouette']:.6f}\n")
            f.write(f"DBI 指数       : {best['topk_min_dbi']:.6f}\n")
            f.write(f"发现于第 {best['iteration']} 轮\n\n")

            f.write("最佳参数:\n")
            for name, val in zip(self.PARAM_NAMES, best['params']):
                f.write(f"  {name:45s}: {val}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("Top 5 稳定评分结果\n")
            f.write("=" * 70 + "\n")
            for rank, r in enumerate(
                    sorted(self.results_log, key=lambda x: x['stability_score'], reverse=True)[:5], 1
            ):
                f.write(f"\n第 {rank} 名 (第{r['iteration']}轮):\n")
                f.write(f"  稳定评分={r['stability_score']:.4f} "
                        f"(综合={r['multi_objective_score']:.4f}±{r['multi_objective_std']:.4f})\n")
                f.write(f"  ACC={r['acc']:.4f}, ARI={r['ari']:.4f}, NMI={r['nmi']:.4f}\n")
                f.write(f"  Sil={r['topk_max_silhouette']:.4f}, DBI={r['topk_min_dbi']:.4f}\n")

        print(f"📝 详细报告已保存: {report_path}")

    def _get_best_result_multi_objective(self):
        if not self.results_log:
            return None
        best = max(self.results_log, key=lambda x: x['stability_score'])
        return {
            'stability_score': best['stability_score'],
            'multi_objective_score': best['multi_objective_score'],
            'acc': best['acc'],
            'ari': best['ari'],
            'nmi': best['nmi'],
            'topk_max_silhouette': best['topk_max_silhouette'],
            'topk_min_dbi': best['topk_min_dbi'],
            'params': dict(zip(self.PARAM_NAMES, best['params'])),
        }


# ================================================================
# ✅ 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='IDC+SEFS 贝叶斯超参数优化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cfg',        type=str, required=True,
                        help='基础配置文件路径（如 cfg/test.yaml）')
    parser.add_argument('--n_calls',    type=int, default=40,
                        help='总采样次数（建议 ≥ 40）')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每批并行评估的参数组数')
    parser.add_argument('--n_workers',  type=int, default=4,
                        help='并行子进程数')
    parser.add_argument('--n_seeds',    type=int, default=3,
                        help='每组参数的随机种子数')
    parser.add_argument('--early_stop', action='store_true',
                        help='启用早停（默认关闭）')
    parser.add_argument('--patience',   type=int,   default=30,
                        help='早停容忍的验证轮数（每 check_val_every_n_epoch 触发一次）')
    parser.add_argument('--min_delta',  type=float, default=0.001,
                        help='早停最小有效提升量')
    parser.add_argument('--monitor',    type=str,
                        default='val/acc_single',
                        help='早停监控指标')
    parser.add_argument('--random_state', type=int, default=42,
                        help='GP 模型随机种子')
    parser.add_argument('--resume',     type=str, default=None,
                        help='断点续跑：指定之前的输出目录路径（如 sefs_bayes_opt_20260727_120000）')
    parser.add_argument('--stability_lambda', type=float, default=0.3,
                        help='方向A：稳定性惩罚系数 λ。优化目标 = 均值 − λ·标准差。'
                             'λ=0 只看均值；λ 越大越偏好低方差（建议 0.3~0.5）')

    args = parser.parse_args()

    # ✅ 断点续跑时，cfg 和 n_calls 等参数从 checkpoint 自动恢复，可不指定
    if args.resume:
        if not os.path.isdir(args.resume):
            print(f"❌ 续跑目录不存在: {args.resume}")
            return
        print(f"🔄 断点续跑模式: {args.resume}")

    # ── 构建早停配置 ──
    early_stop_config = {
        'enabled':   args.early_stop,
        'monitor':   args.monitor,
        'patience':  args.patience,
        'min_delta': args.min_delta,
        'mode':      'max',
        'verbose':   False,
    }

    # ── 初始化并运行优化器 ──
    optimizer = SEFSBayesianOptimizer(
        base_cfg_path   = args.cfg,
        n_calls         = args.n_calls,
        batch_size      = args.batch_size,
        n_workers       = args.n_workers,
        n_seeds         = args.n_seeds,
        random_state    = args.random_state,
        early_stop_config = early_stop_config,
        resume_dir      = args.resume,   # ✅ 断点续跑
        stability_lambda = args.stability_lambda,   # ✅ 方向A：方差惩罚系数
    )

    best_result = optimizer.optimize()

    # ── 打印最终结果 ──
    print("\n" + "=" * 60)
    print("✅ 优化完成！")
    print("=" * 60)
    if best_result:
        print(f"最佳稳定评分: {best_result['stability_score']:.4f} "
              f"(综合={best_result['multi_objective_score']:.4f})")
        print(f"─────────────────────────────────")
        print(f"ACC (准确率)  : {best_result['acc']:.4f}")
        print(f"ARI (调整兰德): {best_result['ari']:.4f}")
        print(f"NMI (标准互信息): {best_result['nmi']:.4f}")
        print(f"Silhouette   : {best_result['topk_max_silhouette']:.4f}")
        print(f"DBI 指数     : {best_result['topk_min_dbi']:.4f}")
        print("\n关键参数建议（直接写入 yaml）:")
        p = best_result['params']
        print(f"  sefs_tau            : {p['sefs_tau']:.3f}")
        print(f"  local_gates_lambda  : {p['local_gates_lambda']:.3f}")
        print(f"  lr_pretrain         : {p['lr_pretrain']:.2e}")


if __name__ == "__main__":
    # ── 多进程启动方式必须为 spawn（避免 CUDA 多进程冲突）──
    mp.set_start_method('spawn', force=True)
    main()
