#!/usr/bin/env python
"""
C-IDC 消融实验自动化脚本。

按实验矩阵生成配置文件并运行实验，记录结果到 CSV。
每个实验在指定数据集上运行多 seed，最终输出汇总表。

用法:
    # 生成所有消融实验的配置文件（不运行）
    python ablation_study.py --mode generate

    # 运行单个实验（指定数据集 + 实验名 + seed）
    python ablation_study.py --mode run --dataset ALLAML --experiment e1_i1_only --seed 0

    # 汇总所有实验结果
    python ablation_study.py --mode summarize --dataset ALLAML
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

# ── 数据集基础配置 ──
BASE_CONFIGS = {
    'ALLAML': {
        'dataset': 'ALLAML',
        'data_file': 'dataset/ALLAML.npz',
        'batch_size': 6,
        'epochs': 1287,
        'ae_non_gated_epochs': 58,
        'ae_pretrain_epochs': 594,
        'start_global_gates_training_on_epoch': 812,
        'mask_percentage': 0.63,
        'latent_noise_std': 0.01,
        'correlation_threshold': 0.075,
        'gtcr_lambda': 0.033,
        'eps': 0.35,
        'gates_hidden_dim': 120,
        'tau': 60,
        'sefs_tau': 1.29,
        'local_gates_lambda': 1.21,
        'global_gates_lambda': 0.0002,
        'lr_pretrain': 8e-4,
        'lr_clustering': 4e-3,
        'lr_aux_classifier': 1e-4,
        'sched_pretrain_min_lr': 2e-5,
        'sched_clustering_min_lr': 5e-7,
    },
    'Prostate': {
        'dataset': 'Prostate',
        'data_file': 'dataset/Prostate.npz',
        'batch_size': 7,
        'epochs': 830,
        'ae_non_gated_epochs': 49,
        'ae_pretrain_epochs': 237,
        'start_global_gates_training_on_epoch': 730,
        'mask_percentage': 0.53,
        'latent_noise_std': 0.01,
        'correlation_threshold': 0.085,
        'gtcr_lambda': 0.084,
        'eps': 0.18,
        'gates_hidden_dim': 122,
        'tau': 41,
        'sefs_tau': 1.24,
        'local_gates_lambda': 1.19,
        'global_gates_lambda': 0.0003,
        'lr_pretrain': 8e-4,
        'lr_clustering': 4e-3,
        'lr_aux_classifier': 1e-4,
        'sched_pretrain_min_lr': 5e-5,
        'sched_clustering_min_lr': 5e-7,
    },
    'SRBCT': {
        'dataset': 'SRBCT',
        'data_file': 'dataset/SRBCT.npz',
        'batch_size': 7,
        'epochs': 733,
        'ae_non_gated_epochs': 44,
        'ae_pretrain_epochs': 202,
        'start_global_gates_training_on_epoch': 473,
        'mask_percentage': 0.44,
        'latent_noise_std': 0.01,
        'correlation_threshold': 0.11,
        'gtcr_lambda': 0.048,
        'eps': 0.18,
        'gates_hidden_dim': 101,
        'tau': 49,
        'sefs_tau': 1.13,
        'local_gates_lambda': 0.87,
        'global_gates_lambda': 0.0001,
        'lr_pretrain': 7e-4,
        'lr_clustering': 3e-3,
        'lr_aux_classifier': 1e-4,
        'sched_pretrain_min_lr': 3e-5,
        'sched_clustering_min_lr': 8e-7,
    },
}

# ── 消融实验矩阵定义 ──
# 每个实验 = 基线配置 + 特定改进开关
ABLATION_EXPERIMENTS = {
    'baseline': {
        'label': 'Baseline',
        'description': '无任何改进（IDC+SEFS 原版）',
        'use_gradual_encdec': False,
        'use_sefs_annealing': False,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'e1_i1_only': {
        'label': 'E1: I1 Only (Gradual EncDec)',
        'description': '仅启用渐进压缩 Encoder-Decoder',
        'use_gradual_encdec': True,
        'use_sefs_annealing': False,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'e2_i2_only': {
        'label': 'E2: I2 Only (SEFS Annealing)',
        'description': '仅启用 SEFS 温度退火',
        'use_gradual_encdec': False,
        'use_sefs_annealing': True,
        'sefs_tau_start': 3.0,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'e3_i3_only': {
        'label': 'E3: I3 Only (K-Means Init)',
        'description': '仅启用 K-Means 聚类头初始化',
        'use_gradual_encdec': False,
        'use_sefs_annealing': False,
        'use_kmeans_init': True,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'e4_i4_only': {
        'label': 'E4: I4 Only (Training Enhancements)',
        'description': '仅启用训练增强（AdamW+Warmup+GradClip+Balance）',
        'use_gradual_encdec': False,
        'use_sefs_annealing': False,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 1e-4,
        'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0.05,
        'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0.1,
        'mixup_alpha': 0,
    },
    'e5_i5_only': {
        'label': 'E5: I5 Only (MixUp)',
        'description': '仅启用 MixUp 数据增强',
        'use_gradual_encdec': False,
        'use_sefs_annealing': False,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0.2,
    },
    'e6_i1_i2': {
        'label': 'E6: I1+I2',
        'description': '渐进压缩 + SEFS 退火',
        'use_gradual_encdec': True,
        'use_sefs_annealing': True,
        'sefs_tau_start': 3.0,
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'e7_i1_i2_i3': {
        'label': 'E7: I1+I2+I3',
        'description': '渐进压缩 + SEFS 退火 + K-Means 初始化',
        'use_gradual_encdec': True,
        'use_sefs_annealing': True,
        'sefs_tau_start': 3.0,
        'use_kmeans_init': True,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    },
    'e8_i1_i2_i3_i4': {
        'label': 'E8: I1+I2+I3+I4',
        'description': '前四个改进（不含 MixUp）',
        'use_gradual_encdec': True,
        'use_sefs_annealing': True,
        'sefs_tau_start': 3.0,
        'use_kmeans_init': True,
        'weight_decay_pretrain': 1e-4,
        'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0.05,
        'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0.1,
        'mixup_alpha': 0,
    },
    'e9_full': {
        'label': 'E9: Full (All 5 Improvements)',
        'description': '全部 5 个改进',
        'use_gradual_encdec': True,
        'use_sefs_annealing': True,
        'sefs_tau_start': 3.0,
        'use_kmeans_init': True,
        'weight_decay_pretrain': 1e-4,
        'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0.05,
        'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0.1,
        'mixup_alpha': 0.2,
    },
    'e10_full_minus_i1': {
        'label': 'E10: Full - I1 (Ablate Architecture)',
        'description': '移除渐进压缩，保留其他 4 个改进',
        'use_gradual_encdec': False,
        'use_sefs_annealing': True,
        'sefs_tau_start': 3.0,
        'use_kmeans_init': True,
        'weight_decay_pretrain': 1e-4,
        'weight_decay_cluster': 1e-5,
        'warmup_ratio': 0.05,
        'grad_clip_norm': 1.0,
        'balance_loss_lambda': 0.1,
        'mixup_alpha': 0.2,
    },
}


def build_cfg(dataset_name, experiment_name):
    """构建消融实验的完整 OmegaConf 配置"""
    base = BASE_CONFIGS[dataset_name]
    exp = ABLATION_EXPERIMENTS[experiment_name]

    cfg_dict = {
        'dataset': base['dataset'],
        'data_file': base['data_file'],
        'batch_size': base['batch_size'],
        'seeds': 10,
        'validate': True,
        'epochs': base['epochs'],
        'ae_non_gated_epochs': base['ae_non_gated_epochs'],
        'ae_pretrain_epochs': base['ae_pretrain_epochs'],
        'start_global_gates_training_on_epoch': base['start_global_gates_training_on_epoch'],
        'mask_percentage': base['mask_percentage'],
        'latent_noise_std': base['latent_noise_std'],
        'correlation_threshold': base['correlation_threshold'],
        'gtcr_loss': True,
        'gtcr_projection_dim': None,
        'gtcr_eps': 1,
        'gtcr_lambda': base['gtcr_lambda'],
        'eps': base['eps'],
        'use_gating': True,
        'gates_hidden_dim': base['gates_hidden_dim'],
        'tau': base['tau'],
        'sefs_tau': base['sefs_tau'],
        'local_gates_lambda': base['local_gates_lambda'],
        'global_gates_lambda': base['global_gates_lambda'],
        'encdec': [512, 512, 2048, 32, 2048, 512, 512],
        'clustering_head': [32, 128],
        'aux_classifier': [128],
        'lr': {
            'pretrain': base['lr_pretrain'],
            'clustering': base['lr_clustering'],
            'aux_classifier': base['lr_aux_classifier'],
        },
        'sched': {
            'pretrain_min_lr': base['sched_pretrain_min_lr'],
            'clustering_min_lr': base['sched_clustering_min_lr'],
        },
        'trainer': {
            'devices': 1,
            'accelerator': 'gpu',
            'max_epochs': base['epochs'],
            'deterministic': True,
            'logger': True,
            'log_every_n_steps': 10,
            'check_val_every_n_epoch': 10,
            'enable_checkpointing': False,
            'num_sanity_val_steps': 0,
        },
        'use_sefs': True,
        'save_seed_checkpoints': False,

        # ── 改进默认值 ──
        'use_gradual_encdec': False,
        'encdec_div_factor': 4,
        'encdec_n_stages': 3,
        'bottleneck_dim': 32,
        'encdec_dropout': 0.2,
        'encdec_skip_connections': True,
        'use_sefs_annealing': False,
        'sefs_tau_start': 3.0,
        'sefs_tau_end': base['sefs_tau'],
        'use_kmeans_init': False,
        'weight_decay_pretrain': 0,
        'weight_decay_cluster': 0,
        'warmup_ratio': 0,
        'grad_clip_norm': 0,
        'balance_loss_lambda': 0,
        'mixup_alpha': 0,
    }

    # ── 应用实验特定开关 ──
    cfg_dict.update(exp)

    return OmegaConf.create(cfg_dict)


def generate_configs(output_dir='cfg/ablation'):
    """生成所有消融实验的配置文件"""
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in BASE_CONFIGS:
        for exp_name, exp_info in ABLATION_EXPERIMENTS.items():
            cfg = build_cfg(dataset_name, exp_name)
            cfg_path = os.path.join(output_dir, f'{dataset_name}_{exp_name}.yaml')
            OmegaConf.save(cfg, cfg_path)
            print(f"✅ 生成: {cfg_path} — {exp_info['label']}")

    print(f"\n全部 {len(BASE_CONFIGS) * len(ABLATION_EXPERIMENTS)} 个配置文件已生成到 {output_dir}/")


def run_experiment(dataset, experiment, seed, cfg_dir='cfg/ablation'):
    """运行单个实验"""
    cfg_path = os.path.join(cfg_dir, f'{dataset}_{experiment}.yaml')
    if not os.path.exists(cfg_path):
        print(f"❌ 配置文件不存在: {cfg_path}")
        print(f"   请先运行: python ablation_study.py --mode generate")
        sys.exit(1)

    cmd = f'python train_evaluate.py --cfg {cfg_path}'
    print(f"🚀 运行: {dataset}/{experiment} seed={seed}")
    print(f"   命令: {cmd}")

    # 设置 seed 环境变量（train_evaluate.py 会循环 cfg.seeds 个 seed）
    env = os.environ.copy()
    subprocess.run(cmd, shell=True, env=env)


def summarize_results(results_dir='.'):
    """汇总 results_train_evaluate.py.txt 文件"""
    import glob
    result_files = glob.glob(os.path.join(results_dir, 'results_*.txt'))
    for rf in result_files:
        print(f"\n{'='*60}")
        print(f"结果文件: {rf}")
        with open(rf) as f:
            content = f.read()
        print(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C-IDC 消融实验工具')
    parser.add_argument('--mode', type=str, default='generate',
                        choices=['generate', 'run', 'summarize'],
                        help='模式: generate=生成配置文件, run=运行实验, summarize=汇总结果')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称 (ALLAML, Prostate, SRBCT)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='实验名称 (baseline, e1_i1_only, ...)')
    parser.add_argument('--seed', type=int, default=0,
                        help='起始 seed')
    parser.add_argument('--cfg-dir', type=str, default='cfg/ablation',
                        help='配置文件输出/读取目录')

    args = parser.parse_args()

    if args.mode == 'generate':
        generate_configs(args.cfg_dir)
        print("\n📋 消融实验矩阵：")
        print(f"{'实验ID':<20} {'标签':<35} {'说明'}")
        print("-" * 90)
        for exp_name, exp_info in ABLATION_EXPERIMENTS.items():
            print(f"{exp_name:<20} {exp_info['label']:<35} {exp_info['description']}")

    elif args.mode == 'run':
        if not args.dataset or not args.experiment:
            print("❌ --mode run 需要指定 --dataset 和 --experiment")
            print("示例: python ablation_study.py --mode run --dataset ALLAML --experiment e9_full")
            sys.exit(1)
        run_experiment(args.dataset, args.experiment, args.seed, args.cfg_dir)

    elif args.mode == 'summarize':
        summarize_results()
