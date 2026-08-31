#!/usr/bin/env python
"""
C-IDC 轻量改进消融实验脚本。

层B实验：11组 × 3数据集 × 10 seeds = 330次训练
通过修改配置开关组合来验证 D1-D7 各改进的贡献。

用法:
    python ablation_lite.py --mode generate     # 生成所有消融配置文件
    python ablation_lite.py --mode run --dataset ALLAML  # 运行指定数据集
    python ablation_lite.py --mode summarize    # 汇总结果
"""
import os
import sys
import argparse
import subprocess
from omegaconf import OmegaConf

# ── 数据集基础配置（与原始 *_lite.yaml 一致）──
BASE = {
    'ALLAML': {
        'dataset': 'ALLAML', 'data_file': 'dataset/ALLAML.npz',
        'batch_size': 6, 'epochs': 1287, 'ae_non_gated_epochs': 58,
        'ae_pretrain_epochs': 594, 'start_global_gates_training_on_epoch': 812,
        'mask_percentage': 0.63, 'correlation_threshold': 0.075,
        'gtcr_lambda': 0.033, 'eps': 0.35, 'gates_hidden_dim': 120,
        'tau': 60, 'sefs_tau': 1.29, 'local_gates_lambda': 1.21,
        'global_gates_lambda': 0.0002, 'sefs_tau_end': 0.8,
        'lr_pretrain': 8e-4, 'lr_clustering': 4e-3,
        'sched_pretrain_min_lr': 2e-5, 'sched_clustering_min_lr': 5e-7,
    },
    'Prostate': {
        'dataset': 'Prostate', 'data_file': 'dataset/Prostate.npz',
        'batch_size': 7, 'epochs': 830, 'ae_non_gated_epochs': 49,
        'ae_pretrain_epochs': 237, 'start_global_gates_training_on_epoch': 730,
        'mask_percentage': 0.53, 'correlation_threshold': 0.085,
        'gtcr_lambda': 0.084, 'eps': 0.18, 'gates_hidden_dim': 122,
        'tau': 41, 'sefs_tau': 1.24, 'local_gates_lambda': 1.19,
        'global_gates_lambda': 0.0003, 'sefs_tau_end': 0.3,
        'lr_pretrain': 8e-4, 'lr_clustering': 4e-3,
        'sched_pretrain_min_lr': 5e-5, 'sched_clustering_min_lr': 5e-7,
    },
    'SRBCT': {
        'dataset': 'SRBCT', 'data_file': 'dataset/SRBCT.npz',
        'batch_size': 7, 'epochs': 733, 'ae_non_gated_epochs': 44,
        'ae_pretrain_epochs': 202, 'start_global_gates_training_on_epoch': 473,
        'mask_percentage': 0.44, 'correlation_threshold': 0.11,
        'gtcr_lambda': 0.048, 'eps': 0.18, 'gates_hidden_dim': 101,
        'tau': 49, 'sefs_tau': 1.13, 'local_gates_lambda': 0.87,
        'global_gates_lambda': 0.0001, 'sefs_tau_end': 0.3,
        'lr_pretrain': 7e-4, 'lr_clustering': 3e-3,
        'sched_pretrain_min_lr': 3e-5, 'sched_clustering_min_lr': 8e-7,
    },
}

# ── 消融实验定义：每个实验覆盖 Base 配置 + 特定开关 ──
EXPERIMENTS = {
    'B0_baseline': {
        'label': 'B0: Baseline (C-IDC default)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': False,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'B1_d1_only': {
        'label': 'B1: D1 Only (Dropout)',
        'encdec_dropout': 0.2,
        'use_sefs_annealing': False, 'use_kmeans_init': False,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0,
    },
    'B2_d2_only': {
        'label': 'B2: D2 Only (AdamW)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': False, 'use_kmeans_init': False,
        'weight_decay_pretrain': 1e-4, 'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0,
    },
    'B3_d3_only': {
        'label': 'B3: D3 Only (SEFS Annealing)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': True, 'use_kmeans_init': False,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0,
    },
    'B4_d4_only': {
        'label': 'B4: D4 Only (GradClip)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': False, 'use_kmeans_init': False,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0,
    },
    'B5_d5_only': {
        'label': 'B5: D5 Only (K-Means Init)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': False, 'use_kmeans_init': True,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0,
    },
    'B6_d6_only': {
        'label': 'B6: D6 Only (MixUp)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': False, 'use_kmeans_init': False,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0.2,
    },
    'B7_d7_only': {
        'label': 'B7: D7 Only (Balance Loss)',
        'encdec_dropout': 0.0,
        'use_sefs_annealing': False, 'use_kmeans_init': False,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0.1, 'mixup_alpha': 0,
    },
    'B8_d1_d3': {
        'label': 'B8: D1+D3 (Dropout + SEFS)',
        'encdec_dropout': 0.2, 'use_sefs_annealing': True,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0,
    },
    'B9_d2_d4_d7': {
        'label': 'B9: D2+D4+D7 (Training)',
        'encdec_dropout': 0.0, 'use_sefs_annealing': False,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 1e-4, 'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0.05, 'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0.1, 'mixup_alpha': 0,
    },
    'B10_d5_d6': {
        'label': 'B10: D5+D6 (K-Means + MixUp)',
        'encdec_dropout': 0.0, 'use_sefs_annealing': False,
        'use_kmeans_init': True,
        'weight_decay_pretrain': 0, 'weight_decay_cluster': 0,
        'warmup_ratio': 0, 'grad_clip_norm': 0,
        'balance_loss_lambda': 0, 'mixup_alpha': 0.2,
    },
    'B11_full_lite': {
        'label': 'B11: Full Lite (D1-D7 All)',
        'encdec_dropout': 0.2,
        'use_sefs_annealing': True, 'use_kmeans_init': True,
        'weight_decay_pretrain': 1e-4, 'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0.05, 'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0.1, 'mixup_alpha': 0.2,
    },
}


def build_cfg(dataset_name, experiment_name):
    """构建消融实验配置"""
    b = BASE[dataset_name]
    e = EXPERIMENTS[experiment_name]

    # SRBCT 特殊处理
    dp = e['encdec_dropout']
    wd_p = e['weight_decay_pretrain']
    wd_c = e['weight_decay_cluster']
    mx = e['mixup_alpha']
    bl = e['balance_loss_lambda']
    if dataset_name == 'SRBCT':
        if dp > 0: dp = 0.3
        if wd_p > 0: wd_p = 2e-4
        if wd_c > 0: wd_c = 2e-5
        if mx > 0: mx = 0.3
        if bl > 0: bl = 0.15

    cfg_dict = {
        'dataset': b['dataset'], 'data_file': b['data_file'],
        'batch_size': b['batch_size'], 'seeds': 1, 'validate': True,
        'epochs': b['epochs'], 'ae_non_gated_epochs': b['ae_non_gated_epochs'],
        'ae_pretrain_epochs': b['ae_pretrain_epochs'],
        'start_global_gates_training_on_epoch': b['start_global_gates_training_on_epoch'],
        'mask_percentage': b['mask_percentage'], 'latent_noise_std': 0.01,
        'correlation_threshold': b['correlation_threshold'],
        'gtcr_loss': True, 'gtcr_projection_dim': None, 'gtcr_eps': 1,
        'gtcr_lambda': b['gtcr_lambda'], 'eps': b['eps'],
        'use_gating': True, 'gates_hidden_dim': b['gates_hidden_dim'],
        'tau': b['tau'], 'sefs_tau': b['sefs_tau'],
        'local_gates_lambda': b['local_gates_lambda'],
        'global_gates_lambda': b['global_gates_lambda'],
        'encdec': [512, 512, 2048, 32, 2048, 512, 512],
        'clustering_head': [32, 128], 'aux_classifier': [128],
        'lr': {'pretrain': b['lr_pretrain'], 'clustering': b['lr_clustering'],
               'aux_classifier': 1e-4},
        'sched': {'pretrain_min_lr': b['sched_pretrain_min_lr'],
                  'clustering_min_lr': b['sched_clustering_min_lr']},
        'trainer': {
            'devices': 1, 'accelerator': 'gpu', 'max_epochs': b['epochs'],
            'deterministic': True, 'logger': True, 'log_every_n_steps': 10,
            'check_val_every_n_epoch': 10, 'enable_checkpointing': False,
            'num_sanity_val_steps': 0,
        },
        'use_sefs': True, 'save_seed_checkpoints': False,
        'sefs_tau_start': 3.0, 'sefs_tau_end': b['sefs_tau_end'],
        # ── 实验特定参数 ──
        'encdec_dropout': dp, 'use_sefs_annealing': e['use_sefs_annealing'],
        'use_kmeans_init': e['use_kmeans_init'],
        'weight_decay_pretrain': wd_p, 'weight_decay_cluster': wd_c,
        'warmup_ratio': e['warmup_ratio'], 'grad_clip_norm': e['grad_clip_norm'],
        'balance_loss_lambda': bl, 'mixup_alpha': mx,
    }
    return OmegaConf.create(cfg_dict)


def generate_configs(output_dir='cfg/ablation_lite'):
    """生成所有消融实验配置文件"""
    os.makedirs(output_dir, exist_ok=True)
    for ds in BASE:
        for exp_name, exp_info in EXPERIMENTS.items():
            cfg = build_cfg(ds, exp_name)
            path = os.path.join(output_dir, f'{ds}_{exp_name}.yaml')
            OmegaConf.save(cfg, path)
            print(f"✅ {path} — {exp_info['label']}")
    print(f"\n共 {len(BASE)*len(EXPERIMENTS)} 个配置文件 → {output_dir}/")


def run_experiments(dataset, cfg_dir='cfg/ablation_lite'):
    """运行指定数据集的所有消融实验（串行，每组 10 seeds）"""
    for exp_name in EXPERIMENTS:
        cfg_path = os.path.join(cfg_dir, f'{dataset}_{exp_name}.yaml')
        if not os.path.exists(cfg_path):
            print(f"  ⚠️ 跳过: {cfg_path} 不存在，先运行 --mode generate")
            continue
        # 临时修改 seeds 为 10
        cfg = OmegaConf.load(cfg_path)
        cfg.seeds = 10
        tmp_path = cfg_path.replace('.yaml', '_run.yaml')
        OmegaConf.save(cfg, tmp_path)
        print(f"\n{'='*60}")
        print(f"🚀 {EXPERIMENTS[exp_name]['label']}")
        print(f"   cfg: {tmp_path}")
        cmd = f'python train_evaluate.py --cfg {tmp_path}'
        subprocess.run(cmd, shell=True)
        os.remove(tmp_path)


def summarize():
    """汇总 results_train_evaluate.py.txt 中的结果"""
    import glob
    for f in sorted(glob.glob('results_*.txt')):
        print(f"\n{'='*60}")
        print(f"📄 {f}")
        with open(f) as fh:
            print(fh.read().strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C-IDC 轻量消融实验')
    parser.add_argument('--mode', choices=['generate', 'run', 'summarize'], default='generate')
    parser.add_argument('--dataset', default=None, help='ALLAML / Prostate / SRBCT')
    parser.add_argument('--cfg-dir', default='cfg/ablation_lite')
    args = parser.parse_args()

    if args.mode == 'generate':
        generate_configs(args.cfg_dir)
        print("\n━━━ 消融实验矩阵 ━━━")
        for eid, einfo in EXPERIMENTS.items():
            print(f"  {eid:<20} {einfo['label']}")

    elif args.mode == 'run':
        if not args.dataset:
            print("❌ 需要 --dataset (ALLAML / Prostate / SRBCT)")
            sys.exit(1)
        run_experiments(args.dataset, args.cfg_dir)

    elif args.mode == 'summarize':
        summarize()
