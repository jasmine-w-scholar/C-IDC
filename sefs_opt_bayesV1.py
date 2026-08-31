# ============================================================
# IDC+SEFS 贝叶斯超参数优化脚本
# 文件名：sefs_bayes_opt.py
# 目标函数是仅探索acc指标
# 使用方式：
#   python sefs_opt_bayes.py  --cfg cfg/test.yaml  --n_calls 40  --batch_size 4  --n_workers 4  --n_seeds 3  --early_stop --patience 30  --min_delta 0.001
#
# 依赖：
#   pip install scikit-optimize pandas pytorch-lightning
# ============================================================

import os
import json
import math
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from skopt import Optimizer
from skopt.space import Real, Integer
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import gc

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

        # ── Step 7: 收集结果 ──
        actual_epochs = trainer.current_epoch
        early_stopped = (
            early_stop_callback is not None and
            early_stop_callback.stopped_epoch > 0
        )

        return {
            'acc':           model.best_acc,
            'ari':           model.best_ari,
            'nmi':           model.best_nmi,
            'actual_epochs': actual_epochs,
            'early_stopped': early_stopped,
            'success':       True,
        }

    except Exception as e:
        print(f"❌ [评估失败] Seed={seed}, 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'acc': 0.0, 'ari': 0.0, 'nmi': 0.0,
            'actual_epochs': 0, 'early_stopped': False, 'success': False,
        }

    finally:
        # ── 显存清理（子进程退出前强制释放）──
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'model'   in dir(): del model    # noqa
        if 'trainer' in dir(): del trainer  # noqa
        gc.collect()


# ================================================================
# ✅ IDC+SEFS 批量贝叶斯优化器
#
# 与 IDC 原版优化器的主要区别：
#   1. 搜索空间新增 sefs_tau（SEFS 温度参数，核心超参数）
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

    # ── 搜索空间定义（含详细注释）──
    #
    # 参数名               范围                    说明
    # ─────────────────────────────────────────────────────
    # epochs               [500, 1000]             总训练轮数
    # ae_non_gated_epochs  [30, 80]                无门控预热轮数
    # ae_pretrain_epochs   [200, 500]              预训练轮数（IDC 特有两阶段结构）
    # start_global_*       [ae_pretrain+50, ~]     全局门控启动轮数
    # gates_hidden_dim     [32, 256]               门控网络隐藏层维度
    # lr_pretrain          [1e-5, 1e-3]            预训练学习率（log 均匀）
    # lr_clustering        [1e-5, 1e-2]            聚类学习率
    # lr_aux_classifier    [1e-5, 1e-2]            辅助分类器学习率
    # sched_pretrain_min_lr [1e-7, 1e-4]           预训练 LR 最小值
    # sched_clustering_min_lr [1e-7, 1e-5]         聚类 LR 最小值
    # local_gates_lambda   [0.5, 5.0]              ⭐ 稀疏正则化权重（SEFS 需要更大）
    # global_gates_lambda  [1e-5, 1e-3]            全局门控正则化权重
    # gtcr_lambda          [0.001, 0.1]            GTCR 损失权重
    # tau                  [10, 200]               Gumbel-Softmax 温度
    # eps                  [0.01, 0.5]             MCRR 精度参数
    # mask_percentage      [0.5, 0.9]              随机 mask 比例
    # sefs_tau             [0.3, 2.0]              ⭐ SEFS 门控温度（核心参数）
    #                                              小→门控更硬（稀疏）
    #                                              大→门控更软（平滑）

    SEARCH_SPACE = [
        # ── 训练周期 ──
        Integer(1000,  2000, name='epochs'),
        Integer(30,   60,   name='ae_non_gated_epochs'),
        Integer(200,  350,  name='ae_pretrain_epochs'),
        Integer(300, 550,  name='start_global_gates_training_on_epoch'),

        # ── 网络架构 ──
        Integer(128,   512,  name='gates_hidden_dim'),

        # ── 学习率（log 均匀分布更合理）──
        Real(1e-3, 5e-3, 'log-uniform', name='lr_pretrain'),
        Real(1e-3, 1e-2, 'log-uniform', name='lr_clustering'),
        Real(1e-4, 1e-3, 'log-uniform', name='lr_aux_classifier'),
        Real(1e-5, 1e-4, 'log-uniform', name='sched_pretrain_min_lr'),
        Real(1e-6, 1e-4, 'log-uniform', name='sched_clustering_min_lr'),

        # ── 损失权重 ──
        # local_gates_lambda 上调至 5.0：
        # SEFS 软门控（m_tilde∈(0,1)）比 IDC 硬门控更"宽松"，
        # 需要更强的 L0 正则化来压制冗余特征
        Real(0.3,  1.0,  name='local_gates_lambda'),
        Real(1e-5, 1e-3, name='global_gates_lambda'),
        Real(0.005, 0.05,  name='gtcr_lambda'),

        # ── 聚类参数 ──
        Real(10.0, 50.0, name='tau'),
        Real(0.05,  0.3,  name='eps'),
        Real(0.2,   0.5,  name='mask_percentage'),

        # ── SEFS 核心超参数 ──
        # sefs_tau 控制 Relaxed-MultiBern 的软化程度：
        #   tau=0.3: 门控接近离散 0/1，竞争更剧烈，可能导致梯度消失
        #   tau=1.0: 门控平滑，梯度流动好，但稀疏性较弱
        #   tau=2.0: 门控极软，几乎无竞争效果
        # 当前 acc≈0.49 的可能原因是 tau=0.8 与 lambda=1.5 的组合仍未收敛
        Real(0.8,  1.5,  name='sefs_tau'),
    ]

    PARAM_NAMES = [
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
        """
        self.base_cfg      = OmegaConf.load(base_cfg_path)
        self.cfg_dict      = OmegaConf.to_container(self.base_cfg, resolve=True)

        self.n_calls       = n_calls
        self.batch_size    = batch_size
        self.n_workers     = min(n_workers, mp.cpu_count())
        self.n_seeds       = n_seeds
        self.random_state  = random_state
        self.n_iterations  = math.ceil(n_calls / batch_size)

        self.early_stop_config = early_stop_config or {'enabled': False}
        self.results_log   = []   # 存储所有评估结果

        # ── 输出目录 ──
        timestamp        = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir  = f"sefs_bayes_opt_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 结果将保存至: {self.output_dir}/")

        # ── 创建高斯过程优化器 ──
        # base_estimator='GP': 使用高斯过程作为代理模型
        # acq_func='EI':       期望改进采集函数（适合有噪声的评估）
        # acq_optimizer='sampling': 通过随机采样优化采集函数（快）
        self.optimizer = Optimizer(
            dimensions=self.SEARCH_SPACE,
            base_estimator='GP',
            acq_func='EI',
            acq_optimizer='sampling',
            random_state=self.random_state,
        )

        self._print_init_info()

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
                    raw_results[pi].append(result)

                    status = "✅" if result['success'] else "❌"
                    es_tag = " [早停]" if result.get('early_stopped') else ""
                    print(f"   {status} 参数组{pi} Seed{task['seed']}: "
                          f"ACC={result['acc']:.4f}{es_tag}")
                except Exception as e:
                    print(f"   ❌ 参数组{pi} Seed{task['seed']} 异常: {e}")
                    if pi not in raw_results:
                        raw_results[pi] = []
                    raw_results[pi].append({
                        'acc': 0., 'ari': 0., 'nmi': 0.,
                        'actual_epochs': 0, 'early_stopped': False, 'success': False
                    })

        # ── 聚合：对每组参数取成功种子的均值 ──
        aggregated = []
        for pi in range(len(params_list)):
            group   = raw_results.get(pi, [])
            success = [r for r in group if r['success']]

            if success:
                agg = {
                    'success':          True,
                    'acc':              np.mean([r['acc']           for r in success]),
                    'ari':              np.mean([r['ari']           for r in success]),
                    'nmi':              np.mean([r['nmi']           for r in success]),
                    'actual_epochs':    np.mean([r['actual_epochs'] for r in success]),
                    'early_stopped':    any(r['early_stopped']      for r in success),
                    'n_successful':     len(success),
                    'n_total':          self.n_seeds,
                }
                print(f"   📊 参数组{pi} 聚合: "
                      f"ACC={agg['acc']:.4f} ± "
                      f"{np.std([r['acc'] for r in success]):.4f} "
                      f"(n={len(success)}/{self.n_seeds})")
            else:
                agg = {
                    'success': False, 'acc': 0., 'ari': 0., 'nmi': 0.,
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
        print(f"🔍 开始优化（共 {self.n_iterations} 轮）\n")

        for iteration in range(1, self.n_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"📍 第 {iteration}/{self.n_iterations} 轮")
            print(f"{'=' * 60}")

            # ── Step 1: GP 推荐候选参数 ──
            candidates = self.optimizer.ask(n_points=self.batch_size)
            print(f"   GP 推荐 {len(candidates)} 组候选参数")

            # ── Step 2: 并行评估 ──
            batch_results = self._evaluate_batch_parallel(candidates)

            # ── Step 3: 将结果告知 GP（最小化负 ACC = 最大化 ACC）──
            for pi, (params, result) in enumerate(zip(candidates, batch_results)):
                if result['success']:
                    loss = -result['acc']           # scikit-optimize 最小化目标
                    self.optimizer.tell(params, loss)

                    # 记录完整结果（含参数信息）
                    self.results_log.append({
                        'iteration':   iteration,
                        'params':      params,
                        'acc':         result['acc'],
                        'ari':         result['ari'],
                        'nmi':         result['nmi'],
                        'actual_epochs': result['actual_epochs'],
                        'early_stopped': result['early_stopped'],
                        'n_successful':  result['n_successful'],
                        'n_total':       result['n_total'],
                    })

            # ── Step 4: 打印当前全局最佳 ──
            if self.results_log:
                best = max(self.results_log, key=lambda x: x['acc'])
                print(f"\n🏆 全局最佳 (第{best['iteration']}轮发现): "
                      f"ACC={best['acc']:.4f}, "
                      f"ARI={best['ari']:.4f}, "
                      f"NMI={best['nmi']:.4f}")
                best_params_dict = dict(zip(self.PARAM_NAMES, best['params']))
                print(f"   sefs_tau={best_params_dict['sefs_tau']:.3f}, "
                      f"local_gates_lambda={best_params_dict['local_gates_lambda']:.3f}, "
                      f"lr_pretrain={best_params_dict['lr_pretrain']:.2e}")

            # ── Step 5: 每轮保存中间结果（即使后续中断也不丢失数据）──
            self._save_intermediate(iteration)

        # ── 最终保存完整报告 ──
        self._save_final_report()
        return self._get_best_result()

    # ────────────────────────────────────────────────────────
    # ✅ 中间结果保存（每轮迭代后调用）
    # ────────────────────────────────────────────────────────
    def _save_intermediate(self, iteration: int):
        """每轮优化结束后立即保存，防止中断丢失数据"""
        if not self.results_log:
            return

        best = max(self.results_log, key=lambda x: x['acc'])

        # ── 写入 TXT（人类可读）──
        txt_path = os.path.join(self.output_dir, f"iter_{iteration:03d}_best.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"第 {iteration} 轮中间结果\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"当前最佳 ACC : {best['acc']:.6f}\n")
            f.write(f"当前最佳 ARI : {best['ari']:.6f}\n")
            f.write(f"当前最佳 NMI : {best['nmi']:.6f}\n")
            f.write(f"发现于第     : {best['iteration']} 轮\n\n")
            f.write("最佳参数:\n")
            for name, val in zip(self.PARAM_NAMES, best['params']):
                f.write(f"  {name:45s}: {val}\n")

        # ── 写入 JSON（编程读取）──
        json_path = os.path.join(self.output_dir, f"iter_{iteration:03d}_best.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'iteration':    iteration,
                'timestamp':    datetime.now().isoformat(),
                'best_acc':     float(best['acc']),
                'best_ari':     float(best['ari']),
                'best_nmi':     float(best['nmi']),
                'found_at_iter': int(best['iteration']),
                'params':       dict(zip(self.PARAM_NAMES,
                                         [float(v) for v in best['params']])),
            }, f, indent=4)

        print(f"   💾 中间结果已保存: {txt_path}")

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
                'iteration':     r['iteration'],
                'acc':           r['acc'],
                'ari':           r['ari'],
                'nmi':           r['nmi'],
                'actual_epochs': r['actual_epochs'],
                'early_stopped': r['early_stopped'],
                'n_successful':  r['n_successful'],
                'n_total':       r['n_total'],
            }
            row.update(dict(zip(self.PARAM_NAMES, r['params'])))
            rows.append(row)

        df = pd.DataFrame(rows).sort_values('acc', ascending=False)
        csv_path = os.path.join(self.output_dir, 'all_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n📊 完整结果已保存: {csv_path}")

        # ── 2. 最佳参数 JSON ──
        best = max(self.results_log, key=lambda x: x['acc'])
        best_json_path = os.path.join(self.output_dir, 'best_params.json')
        with open(best_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_acc':   float(best['acc']),
                'best_ari':   float(best['ari']),
                'best_nmi':   float(best['nmi']),
                'found_at_iteration': int(best['iteration']),
                'params': dict(zip(self.PARAM_NAMES,
                                   [float(v) for v in best['params']])),
            }, f, indent=4, ensure_ascii=False)
        print(f"🏆 最佳参数已保存: {best_json_path}")

        # ── 3. 人类可读报告 TXT ──
        report_path = os.path.join(self.output_dir, 'optimization_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("IDC+SEFS 贝叶斯优化完整报告\n")
            f.write("=" * 70 + "\n")
            f.write(f"生成时间       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总采样次数     : {self.n_calls}\n")
            f.write(f"每批参数组数   : {self.batch_size}\n")
            f.write(f"每组种子数     : {self.n_seeds}\n")
            f.write(f"成功评估参数组 : {len(self.results_log)}\n\n")

            f.write("=" * 70 + "\n")
            f.write("最佳结果\n")
            f.write("=" * 70 + "\n")
            f.write(f"ACC  : {best['acc']:.6f}\n")
            f.write(f"ARI  : {best['ari']:.6f}\n")
            f.write(f"NMI  : {best['nmi']:.6f}\n")
            f.write(f"发现于第 {best['iteration']} 轮\n\n")

            f.write("最佳参数:\n")
            for name, val in zip(self.PARAM_NAMES, best['params']):
                f.write(f"  {name:45s}: {val}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("Top 5 结果\n")
            f.write("=" * 70 + "\n")
            for rank, r in enumerate(
                sorted(self.results_log, key=lambda x: x['acc'], reverse=True)[:5], 1
            ):
                f.write(f"\n第 {rank} 名 (第{r['iteration']}轮):\n")
                f.write(f"  ACC={r['acc']:.4f}, ARI={r['ari']:.4f}, NMI={r['nmi']:.4f}\n")
                best_p = dict(zip(self.PARAM_NAMES, r['params']))
                f.write(f"  sefs_tau={best_p['sefs_tau']:.3f}, "
                        f"local_gates_lambda={best_p['local_gates_lambda']:.3f}, "
                        f"lr_pretrain={best_p['lr_pretrain']:.2e}\n")

        print(f"📝 详细报告已保存: {report_path}")

    def _get_best_result(self):
        if not self.results_log:
            return None
        best = max(self.results_log, key=lambda x: x['acc'])
        return {
            'acc':    best['acc'],
            'ari':    best['ari'],
            'nmi':    best['nmi'],
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

    args = parser.parse_args()

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
    )

    best_result = optimizer.optimize()

    # ── 打印最终结果 ──
    print("\n" + "=" * 60)
    print("✅ 优化完成！")
    print("=" * 60)
    if best_result:
        print(f"最佳 ACC : {best_result['acc']:.4f}")
        print(f"最佳 ARI : {best_result['ari']:.4f}")
        print(f"最佳 NMI : {best_result['nmi']:.4f}")
        print("\n关键参数建议（直接写入 yaml）:")
        p = best_result['params']
        print(f"  sefs_tau            : {p['sefs_tau']:.3f}")
        print(f"  local_gates_lambda  : {p['local_gates_lambda']:.3f}")
        print(f"  lr_pretrain         : {p['lr_pretrain']:.2e}")
        print(f"  ae_pretrain_epochs  : {int(p['ae_pretrain_epochs'])}")
        print(f"  mask_percentage     : {p['mask_percentage']:.2f}")


if __name__ == "__main__":
    # ── 多进程启动方式必须为 spawn（避免 CUDA 多进程冲突）──
    mp.set_start_method('spawn', force=True)
    main()
