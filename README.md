# C-IDC: Enhanced Deep Clustering Based on Collaborative Feature Competition for Tabular Genetic Data

C-IDC 是一个面向小样本高维数据的深度聚类方法，在 IDC框架基础上，引入**特征相关性建模**的门控机制（SEFS 相关噪声），实现更有效的特征选择。

## 环境配置

```bash
# 建议使用 conda
conda create -n cidc python=3.12
conda activate cidc

# 安装 torch
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install -r requirements.txt
```

## 数据集

三个公开的基因表达数据集，预处理后的 `.npz` 文件位于 `dataset/` 目录：

| 数据集 | 样本数 | 特征数 | 类别数 |
|--------|:---:|:---:|:---:|
| ALLAML | 72 | 7129 | 2 |
| Prostate | 102 | 5966 | 2 |
| SRBCT | 83 | 2308 | 4 |

## 复现结果

各数据集的最优配置已保存在 `cfg/` 目录。直接运行：

```bash
# ALLAML（ACC 均值约 0.800）
PYTHONIOENCODING=utf-8 python train_evaluate.py --cfg cfg/ALLAML_best.yaml

# Prostate（ACC 均值约 0.673）
PYTHONIOENCODING=utf-8 python train_evaluate.py --cfg cfg/Prostate_best.yaml

# SRBCT（ACC 均值约 0.555）
PYTHONIOENCODING=utf-8 python train_evaluate.py --cfg cfg/SRBCT_best.yaml
```

> Windows 下需加 `PYTHONIOENCODING=utf-8`（避免 GBK 编码错误）。

## 目录结构

```
CIDC/
├── model.py                # 模型定义
├── train_evaluate.py       # 训练/评估主脚本
├── dataset.py              # 数据加载 + 特征相关性预处理（OAS + Cholesky）
├── sefs_bayes_opt.py       # 贝叶斯超参数搜索
├── cfg/                    # 配置文件
│   ├── ALLAML.yaml    
└── dataset/                # 数据集
```

## 引用

如使用本代码，请引用对应的论文（投稿中）。
