# ============================================================
# sefs_tau 单参数敏感性分析脚本
# 文件名：sensitivity_sefs_tau.py
#
# 项目目录结构假设（与当前 C-IDC 项目一致）：
#   CIDC/
#   ├── sensitivity_sefs_tau.py   ← 本脚本
#   ├── train_evaluate.py
#   ├── model.py
#   ├── dataset.py
#   ├── cfg/
#   │   ├── ALLAML.yaml
#   │   ├── Prostate.yaml
#   │   ├── SRBCT.yaml
#   │   └── 模拟数据集.yaml
#   └── dataset/
#       ├── ALLAML.npz
#       ├── correlate_data.npz
#       ├── Prostate.npz
#       └── SRBCT.npz
#
# 功能：
#   固定其他参数为各数据集贝叶斯优化后的最优值（来自对应 yaml），
#   仅扫描 sefs_tau 在候选值集合上的性能变化，
#   输出 ACC/NMI 均值±标准差折线图（PNG + PDF）及汇总 CSV。
#
# 调用方式（完整4数据集）：
# python sensitivity_sefs_tau.py --n_seeds 10

#python sensitivity_sefs_tau.py --n_seeds 5 --datasets prostate amlall srbct
#
# 调用方式（快速调试，只跑1个数据集）：
#   python sensitivity_sefs_tau.py --datasets prostate --n_seeds 2 --tau_range 0.5 1.56 3.0
# ============================================================

import os
import sys
import argparse
import json
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # 非交互式后端，适合 Windows 无显示器服务器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

# ── 从项目中导入 BaseModule（确保本脚本与 train_evaluate.py 同目录）──
from train_evaluate import BaseModule

# ============================================================
# ✅ 全局配置：数据集信息
# ============================================================

# 数据集键名 → (yaml配置路径, 显示名称)
# 注意：yaml 中已包含 data_file 字段指向数据集路径，无需在此重复指定
DATASET_CONFIG = {
    'syn':      ('cfg/模拟数据集.yaml', 'Synthetic'),
    'prostate': ('cfg/Prostate.yaml',   'Prostate'),
    'amlall':   ('cfg/ALLAML.yaml',     'AMLALL'),
    'srbct':    ('cfg/SRBCT.yaml',      'SRBCT'),
}

# 折线图颜色（与数据集一一对应，参考 matplotlib 默认色板）
DATASET_COLORS = {
    'syn':      '#1f77b4',   # 蓝色
    'prostate': '#ff7f0e',   # 橙色
    'amlall':   '#2ca02c',   # 绿色
    'srbct':    '#d62728',   # 红色
}

# 折线图标记符号
DATASET_MARKERS = {
    'syn':      'o',    # 圆形
    'prostate': 's',    # 方形
    'amlall':   '^',    # 三角形
    'srbct':    'D',    # 菱形
}

# ── sefs_tau 扫描范围 ──
# 覆盖从"门控极硬"(0.3)到"门控极软"(3.0)的完整区间
# 当前贝叶斯优化最优值 1.56 包含在内（用于对比）
DEFAULT_TAU_RANGE = [0.3, 0.5, 0.8, 1.0, 1.2, 1.3,1.4,1.5,1.8, 2.0, 3.0]

# ============================================================
# ✅ 单次训练评估函数
# ============================================================

def run_single_experiment(cfg, seed: int) -> dict:
    """
    使用指定配置和随机种子运行一次完整训练，返回评估指标。

    与 train_evaluate.py 主循环的区别：
    - 禁用 TensorBoard 日志（加速 I/O，避免生成大量日志文件）
    - 禁用 checkpoint 保存
    - 不写入 results_*.txt 文件

    参数：
        cfg  - OmegaConf 配置对象（已设置好所有超参数）
        seed - 随机种子整数

    返回：
        dict: {
            'acc':        float,   # 最佳聚类准确率
            'nmi':        float,   # 最佳 NMI
            'ari':        float,   # 最佳 ARI
            'open_gates': float,   # 开放门控平均数量（特征选择稀疏性指标）
            'success':    bool     # 是否成功完成（False 表示训练过程出现异常）
        }
    """
    try:
        # ── 设置随机种子（与 train_evaluate.py 中保持一致）──
        cfg.seed = seed
        seed_everything(seed, workers=True)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ── 复制配置，禁用日志和检查点（避免污染实验目录）──
        cfg_run = cfg.copy()
        cfg_run.trainer.logger             = False
        cfg_run.trainer.enable_checkpointing = False

        # ── 初始化模型（与 train_evaluate.py 完全一致）──
        model = BaseModule(cfg_run)

        # ── 初始化 Trainer（不添加额外回调）──
        trainer = Trainer(
            **cfg_run.trainer,
            callbacks=[]    # 敏感性分析不使用早停，确保每个 tau 值训练完整轮数
        )

        # ── 训练 ──
        trainer.fit(model)

        # ── 提取 BaseModule 中记录的最佳指标 ──
        # 对应 train_evaluate.py 中的 update_stats() 方法
        acc        = float(model.best_acc)  if hasattr(model, 'best_acc')         else 0.0
        nmi        = float(model.best_nmi)  if hasattr(model, 'best_nmi')         else 0.0
        ari        = float(model.best_ari)  if hasattr(model, 'best_ari')         else 0.0
        open_gates = float(model.best_local_feats) \
                     if hasattr(model, 'best_local_feats') and \
                        model.best_local_feats is not None else 0.0

        print(f"      ✅ Seed={seed}: ACC={acc:.4f}, NMI={nmi:.4f}, "  
              f"ARI={ari:.4f}, OpenGates={open_gates:.2f}")

        # ── 释放 GPU 显存（RTX 5060 共 8GB，多次训练需及时释放）──
        del model, trainer
        torch.cuda.empty_cache()

        return {
            'acc': acc, 'nmi': nmi, 'ari': ari,
            'open_gates': open_gates, 'success': True
        }

    except Exception as e:
        print(f"      ❌ Seed={seed} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {
            'acc': 0.0, 'nmi': 0.0, 'ari': 0.0,
            'open_gates': 0.0, 'success': False
        }

# ============================================================
# ✅ 单数据集的 sefs_tau 参数扫描
# ============================================================

def scan_sefs_tau_single_dataset(
        cfg_path:    str,
        dataset_key: str,
        tau_range:   list,
        n_seeds:     int,
        output_dir:  str
) -> dict:
    """
    对单个数据集执行 sefs_tau 单参数扫描。

    设计说明：
        - 从 cfg_path 加载该数据集的最优参数配置（贝叶斯优化结果）
        - 只修改 sefs_tau，其余参数保持 yaml 中的值不变
        - 每个 tau 值运行 n_seeds 次，取均值±标准差
        - 每个 tau 值完成后立即写入 JSON（防止中途崩溃丢失数据）

    参数：
        cfg_path    - yaml 配置文件路径（包含该数据集的最优参数）
        dataset_key - 数据集标识符（用于日志和文件命名）
        tau_range   - sefs_tau 候选值列表
        n_seeds     - 每个 tau 值运行的随机种子数
        output_dir  - 结果保存目录

    返回：
        {
            tau_val (float): {
                'acc_mean':  float,
                'acc_std':   float,
                'nmi_mean':  float,
                'nmi_std':   float,
                'ari_mean':  float,
                'ari_std':   float,
                'open_gates_mean': float,
                'raw_accs':  list,
                'raw_nmis':  list,
                'raw_aris':  list,
                'n_success': int,
                'n_total':   int,
            }
        }
    """
    display_name = DATASET_CONFIG[dataset_key][1]
    print(f"\n{'='*65}")
    print(f"📊 数据集: {display_name}  |  配置: {cfg_path}")
    print(f"   sefs_tau 扫描范围: {tau_range}")
    print(f"   每个 tau 值种子数: {n_seeds}")
    print(f"{'='*65}")

    # ── 加载该数据集的最优参数配置 ──
    base_cfg = OmegaConf.load(cfg_path)

    # ── 强制启用 SEFS（敏感性分析必须在 SEFS 模式下进行）──
    # 对应 cfg_syn_c1.yaml 中的 use_sefs: true
    base_cfg.use_sefs = True

    results_dict = {}

    for tau_val in tau_range:
        print(f"\n  🔍 sefs_tau = {tau_val:.3f}")

        # ── 关键：只修改 sefs_tau，其他参数保持 yaml 中的最优值 ──
        cfg = base_cfg.copy()
        cfg.sefs_tau = float(tau_val)

        accs, nmis, aris, gates_list = [], [], [], []

        for seed in range(n_seeds):
            result = run_single_experiment(cfg, seed)
            if result['success']:
                accs.append(result['acc'])
                nmis.append(result['nmi'])
                aris.append(result['ari'])
                gates_list.append(result['open_gates'])

        # ── 统计聚合 ──
        n_success = len(accs)
        if n_success > 0:
            tau_result = {
                'acc_mean':        float(np.mean(accs)),
                'acc_std':         float(np.std(accs)),
                'nmi_mean':        float(np.mean(nmis)),
                'nmi_std':         float(np.std(nmis)),
                'ari_mean':        float(np.mean(aris)),
                'ari_std':         float(np.std(aris)),
                'open_gates_mean': float(np.mean(gates_list)),
                'raw_accs':        accs,
                'raw_nmis':        nmis,
                'raw_aris':        aris,
                'n_success':       n_success,
                'n_total':         n_seeds,
            }
            print(f"  📈 tau={tau_val:.3f}: "  
                  f"ACC={tau_result['acc_mean']:.4f}±{tau_result['acc_std']:.4f}  "  
                  f"NMI={tau_result['nmi_mean']:.4f}±{tau_result['nmi_std']:.4f}  "  
                  f"OpenGates={tau_result['open_gates_mean']:.2f}  "  
                  f"[{n_success}/{n_seeds} 种子成功]")
        else:
            # 所有种子均失败，填充零值并记录
            print(f"  ❌ tau={tau_val:.3f}: 所有 {n_seeds} 个种子均失败，填充零值")
            tau_result = {
                'acc_mean': 0.0, 'acc_std': 0.0,
                'nmi_mean': 0.0, 'nmi_std': 0.0,
                'ari_mean': 0.0, 'ari_std': 0.0,
                'open_gates_mean': 0.0,
                'raw_accs': [], 'raw_nmis': [], 'raw_aris': [],
                'n_success': 0, 'n_total': n_seeds,
            }

        results_dict[tau_val] = tau_result

        # ── 每个 tau 值完成后立即保存（防止训练中途崩溃丢失数据）──
        _save_partial_json(results_dict, dataset_key, output_dir)

    # ── 保存该数据集最终完整 JSON ──
    json_path = os.path.join(output_dir, f'raw_{dataset_key}.json')
    _save_dict_to_json(results_dict, json_path)
    print(f"\n  💾 {display_name} 完整结果已保存: {json_path}")

    return results_dict

def _save_partial_json(results_dict: dict, dataset_key: str, output_dir: str):
    """每个 tau 值完成后立即写入 JSON，防止崩溃丢失数据"""
    json_path = os.path.join(output_dir, f'partial_{dataset_key}.json')
    _save_dict_to_json(results_dict, json_path)

def _save_dict_to_json(d: dict, path: str):
    """将 float 键的字典序列化为 JSON（float 键需转为字符串）"""
    str_key_dict = {str(k): v for k, v in d.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(str_key_dict, f, indent=4, ensure_ascii=False)

# ============================================================
# ✅ 绘图函数：ACC 和 NMI 折线图
# ============================================================

def plot_sensitivity_line(
        all_results:  dict,
        tau_range:    list,
        output_dir:   str,
        metric:       str = 'acc',          # 'acc' 或 'nmi'
):
    """
    绘制 sefs_tau 单参数敏感性折线图。

    图形规格（参考 NeurIPS/ICML 论文标准）：
    - 横轴：sefs_tau 值
    - 纵轴：ACC 或 NMI 均值
    - 误差棒：±1σ（种子间标准差）
    - 多条曲线：4个数据集，颜色+标记区分
    - 同时输出 PNG（300dpi）和 PDF（矢量图，论文投稿用）

    参数：
        all_results  - {dataset_key: {tau_val: {...}}} 嵌套字典
        tau_range    - sefs_tau 候选值列表（横轴刻度）
        output_dir   - 保存目录
        metric       - 'acc' 或 'nmi'
        optimal_tau  - 保留参数以兼容接口（不再使用）
    """
    metric_key_mean = f'{metric}_mean'
    metric_key_std  = f'{metric}_std'
    metric_label    = 'ACC' if metric == 'acc' else 'NMI'

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── 绘制每个数据集的折线 ──
    for dataset_key, tau_results in all_results.items():
        display_name = DATASET_CONFIG[dataset_key][1]
        color        = DATASET_COLORS[dataset_key]
        marker       = DATASET_MARKERS[dataset_key]

        # 只取成功计算了的 tau 值（n_success > 0）
        valid_taus  = [t for t in tau_range
                       if t in tau_results and tau_results[t]['n_success'] > 0]
        means       = [tau_results[t][metric_key_mean] for t in valid_taus]
        stds        = [tau_results[t][metric_key_std]  for t in valid_taus]

        if not valid_taus:
            print(f"  ⚠️  {display_name} 无有效结果，跳过绘图")
            continue

        ax.errorbar(
            valid_taus, means,
            yerr=stds,
            label=display_name,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=1.8,
            capsize=4,        # 误差棒端帽长度
            capthick=1.2,
            elinewidth=1.0,
            zorder=3,
        )

    # ── 坐标轴设置 ──
    ax.set_xlabel(r'$\tau_{\mathrm{sefs}}$', fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_xticks(tau_range)
    # 格式化横轴刻度：小于1的用小数，大于等于1的用整数或一位小数
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x:.2g}')
    )
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(bottom=0.0)

    # ── 图例（自动选择最佳位置）──
    ax.legend(
        fontsize=10,
        loc='best',
        framealpha=0.9,
        edgecolor='lightgray',
    )

    # ── 去除顶部和右侧边框（顶会简洁风格）──
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── 横向网格线（辅助读数）──
    ax.yaxis.grid(True, linestyle=':', alpha=0.5, color='gray', zorder=0)

    plt.tight_layout()

    # ── 保存 PNG（演示用）和 PDF（论文投稿用矢量图）──
    base_name = f'sensitivity_sefs_tau_{metric}'
    png_path  = os.path.join(output_dir, f'{base_name}.png')
    pdf_path  = os.path.join(output_dir, f'{base_name}.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"  📊 {metric_label} 折线图已保存:")
    print(f"     PNG: {png_path}")
    print(f"     PDF: {pdf_path}")

# ============================================================
# ✅ 保存汇总 CSV
# ============================================================

def save_summary_csv(
        all_results: dict,
        tau_range:   list,
        output_dir:  str
):
    """
    将所有数据集、所有 tau 值的结果汇总为 CSV 文件。
    CSV 可直接用 Excel 打开查看，也可用于后续自定义绘图。

    列说明：
        dataset          - 数据集名称
        sefs_tau         - 当前扫描的 tau 值
        acc_mean/std     - ACC 均值和标准差
        nmi_mean/std     - NMI 均值和标准差
        ari_mean/std     - ARI 均值和标准差
        open_gates_mean  - 平均开放门控数（特征稀疏性指标）
        n_success        - 成功运行的种子数
        n_total          - 总种子数
    """
    csv_path   = os.path.join(output_dir, 'sensitivity_sefs_tau_summary.csv')
    fieldnames = [
        'dataset', 'sefs_tau',
        'acc_mean', 'acc_std',
        'nmi_mean', 'nmi_std',
        'ari_mean', 'ari_std',
        'open_gates_mean',
        'n_success', 'n_total',
    ]

    rows = []
    for dataset_key, tau_results in all_results.items():
        display_name = DATASET_CONFIG[dataset_key][1]
        for tau_val in tau_range:
            if tau_val not in tau_results:
                continue
            r = tau_results[tau_val]
            rows.append({
                'dataset':         display_name,
                'sefs_tau':        tau_val,
                'acc_mean':        round(r['acc_mean'],        6),
                'acc_std':         round(r['acc_std'],         6),
                'nmi_mean':        round(r['nmi_mean'],        6),
                'nmi_std':         round(r['nmi_std'],         6),
                'ari_mean':        round(r['ari_mean'],        6),
                'ari_std':         round(r['ari_std'],         6),
                'open_gates_mean': round(r['open_gates_mean'], 4),
                'n_success':       r['n_success'],
                'n_total':         r['n_total'],
            })

    # utf-8-sig：带 BOM 的 UTF-8，Excel 直接打开中文不乱码
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n📋 汇总 CSV 已保存: {csv_path}")

# ============================================================
# ✅ 终端打印汇总表格
# ============================================================

def print_summary_table(all_results: dict, tau_range: list):
    """在终端以表格形式打印所有数据集、所有 tau 值的 ACC 结果"""
    print("\n" + "="*75)
    print("📊 sefs_tau 敏感性分析汇总（ACC 均值 ± 标准差）")
    print("="*75)

    dataset_keys = list(all_results.keys())
    # 表头
    header = f"{'tau':>8}"
    for dk in dataset_keys:
        name = DATASET_CONFIG[dk][1]
        header += f"  {name:>18}"
    print(header)
    print("-"*75)

    for tau_val in tau_range:
        row = f"{tau_val:>8.3f}"
        for dk in dataset_keys:
            tau_results = all_results[dk]
            if tau_val in tau_results and tau_results[tau_val]['n_success'] > 0:
                r = tau_results[tau_val]
                row += f"  {r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
            else:
                row += f"  {'N/A':>18}"
        print(row)

    print("="*75)

# ============================================================
# ✅ 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='C-IDC sefs_tau 单参数敏感性分析',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── 数据集选择（默认全部4个）──
    parser.add_argument(
        '--datasets', type=str, nargs='+',
        default=['syn', 'prostate', 'amlall', 'srbct'],
        choices=['syn', 'prostate', 'amlall', 'srbct'],
        help='参与分析的数据集，可指定子集（如 --datasets prostate srbct）'
    )

    # ── sefs_tau 扫描范围（默认值已覆盖完整区间）──
    parser.add_argument(
        '--tau_range', type=float, nargs='+',
        default=DEFAULT_TAU_RANGE,
        help='sefs_tau 候选值列表（空格分隔）'
    )

    # ── 种子数量（建议5，完整实验建议10，调试建议2）──
    parser.add_argument(
        '--n_seeds', type=int, default=5,
        help='每个 tau 值运行的随机种子数（建议5，调试用2）'
    )


    # ── 结果保存目录 ──
    parser.add_argument(
        '--output_dir', type=str,
        default='sensitivity_results_sefs_tau',
        help='结果保存目录（自动创建）'
    )

    args = parser.parse_args()

    # ── 创建输出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("🚀 C-IDC sefs_tau 单参数敏感性分析")
    print("="*65)
    print(f"  参与数据集     : {args.datasets}")
    print(f"  sefs_tau 范围  : {args.tau_range}")
    print(f"  每个值种子数   : {args.n_seeds}")
    print(f"  最优 tau 标注  : {args.optimal_tau}")
    print(f"  结果保存目录   : {args.output_dir}/")
    print("="*65)

    # ── 检查 yaml 配置文件是否存在 ──
    print("\n🔎 检查配置文件...")
    valid_datasets = []
    for key in args.datasets:
        cfg_path, display_name = DATASET_CONFIG[key]
        if os.path.exists(cfg_path):
            print(f"  ✅ [{display_name}]  {cfg_path}")
            valid_datasets.append(key)
        else:
            print(f"  ❌ [{display_name}]  {cfg_path}  ← 文件不存在，跳过")

    if not valid_datasets:
        print("\n❌ 所有配置文件均不存在，程序退出。")
        print("   请确认 cfg/ 目录下存在以下文件：")
        for key in args.datasets:
            print(f"     {DATASET_CONFIG[key][0]}")
        sys.exit(1)

    # ── GPU 信息确认 ──
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        print(f"\n✅ GPU 已就绪: {gpu_name}（显存: {gpu_mem} MiB）")
    else:
        print("\n⚠️  未检测到 GPU，将使用 CPU 训练（速度较慢）")
        print("   如需使用 GPU，请检查 CUDA 安装与 PyTorch 版本匹配")

    # ── 全局确定性设置（与 train_evaluate.py 保持一致）──
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ============================================================
    # 主循环：逐数据集扫描
    # ============================================================
    all_results = {}

    for dataset_key in valid_datasets:
        cfg_path, display_name = DATASET_CONFIG[dataset_key]

        tau_results = scan_sefs_tau_single_dataset(
            cfg_path    = cfg_path,
            dataset_key = dataset_key,
            tau_range   = args.tau_range,
            n_seeds     = args.n_seeds,
            output_dir  = args.output_dir,
        )
        all_results[dataset_key] = tau_results

    # ── 打印终端汇总表格 ──
    print_summary_table(all_results, args.tau_range)

    # ── 保存汇总 CSV ──
    save_summary_csv(all_results, args.tau_range, args.output_dir)

    # ── 保存全量 JSON（含每种子原始数据）──
    all_json_path = os.path.join(args.output_dir, 'all_results.json')
    all_results_str = {
        dk: {str(tv): tr for tv, tr in dr.items()}
        for dk, dr in all_results.items()
    }
    with open(all_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_str, f, indent=4, ensure_ascii=False)
    print(f"💾 全量结果 JSON 已保存: {all_json_path}")

    # ── 绘制 ACC 折线图 ──
    print("\n🎨 正在生成 ACC 折线图...")
    plot_sensitivity_line(
        all_results = all_results,
        tau_range   = args.tau_range,
        output_dir  = args.output_dir,
        metric      = 'acc'
    )

    # ── 最终输出文件清单 ──
    print("\n" + "="*65)
    print("✅ sefs_tau 敏感性分析完成！")
    print(f"   结果目录: {args.output_dir}/")
    print("   输出文件清单:")
    for fname in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, fname)
        fsize = os.path.getsize(fpath)
        print(f"     {fname:<50} ({fsize:>8} bytes)")
    print("="*65)

# ============================================================
# ✅ 程序入口
# ============================================================
if __name__ == '__main__':
    # Windows 下多进程必须使用 spawn 方式（PyTorch 要求）
    # 即使本脚本不使用多进程，保留此设置避免子模块触发 fork 错误
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()