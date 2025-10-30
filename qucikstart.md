# PGPortfolio 快速开始指南

本指南将帮助你快速配置环境并运行 PGPortfolio (PyTorch 版本)。

## 📋 前置要求

- Python 3.11 或更高版本
- uv (推荐) 或 pip
- Git

## 🚀 环境配置

### 方法 1：使用 uv (推荐)

[uv](https://github.com/astral-sh/uv) 是一个极快的 Python 包管理器，推荐使用。

#### 1. 安装 uv

使用 pip 安装：
```bash
pip install uv
```

#### 2. 克隆项目并配置环境

```bash
# 克隆项目
git clone 
cd PGPortfolio

# 使用 uv 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 安装项目依赖
uv pip install -e .

# 或者直接从 requirements.txt 安装
uv pip install -r requirements.txt
```

### 方法 2：使用传统 pip + venv

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 方法 3：使用 conda

```bash
# 创建 conda 环境
conda create -n pgportfolio python=3.11
conda activate pgportfolio

# 安装 PyTorch (根据你的系统选择)
# CPU 版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# GPU 版本 (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

## 📦 依赖项说明

核心依赖：
- `torch>=2.0.0` - PyTorch 深度学习框架
- `numpy>=1.21.0` - 数值计算
- `pandas>=1.3.0` - 数据处理
- `tensorboard>=2.0.0` - 训练可视化
- `cvxopt>=1.3.0` - 凸优化（用于传统策略）

## 🎯 快速运行

### 1. 下载数据（首次运行必需）

```bash
python main.py --mode=download_data
```

这将从 Poloniex 下载历史价格数据到 `./database/` 目录。

或者自行复制到目录。
### 2. 生成训练配置

```bash
python main.py --mode=generate --repeat=1
```

这会在 `./train_package/` 目录下创建训练文件夹（如 `1/`），包含：
- `net_config.json` - 网络配置文件
- 其他训练所需的文件

**参数说明：**
- `--repeat=N` - 生成 N 个不同的训练配置

### 3. 训练模型

#### CPU 训练
```bash
python main.py --mode=train --processes=1 --device=cpu
```

#### GPU 训练
```bash
python main.py --mode=train --processes=1 --device=cuda
```

**参数说明：**
- `--processes=N` - 并行训练进程数（建议从 1 开始）
- `--device=cpu|cuda` - 使用 CPU 或 GPU 训练

**训练输出：**
训练完成后，模型和日志会保存在 `./train_package/1/` 目录：
```
train_package/1/
├── netfile                 # PyTorch 模型权重
├── net_config.json         # 网络配置
├── tensorboard/            # TensorBoard 日志
├── programlog              # 训练日志
└── train_summary.csv       # 训练摘要
```

### 4. 查看训练过程（可选）

在训练过程中或训练后，可以使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir=./train_package/1/tensorboard --port=6006
```

然后在浏览器中打开 http://localhost:6006

### 5. 回测模型

```bash
python main.py --mode=backtest --algo=1
```

**参数说明：**
- `--algo=N` - 指定要回测的训练文件夹编号（如 1, 2, 3）

### 6. 绘制结果对比图

```bash
python main.py --mode=plot --algos=crp,olmar,1
```

**参数说明：**
- `--algos` - 要比较的算法，用逗号分隔
  - `crp` - Constant Rebalanced Portfolio (均匀投资)
  - `olmar` - Online Moving Average Reversion
  - `1`, `2`, `3` - 你训练的模型编号

可选参数：
```bash
python main.py --mode=plot --algos=crp,olmar,1 --labels="均匀策略,OLMAR,我的模型"
```

### 7. 生成结果表格

```bash
python main.py --mode=table --algos=crp,olmar,1 --format=raw
```

**format 选项：**
- `raw` - 原始格式
- `latex` - LaTeX 表格格式
- `csv` - CSV 格式

## 🔧 配置文件说明

主要配置文件：`./pgportfolio/net_config.json`

```json
{
  "input": {
    "feature_number": 3,        // 特征数量（开盘、最高、最低价）
    "coin_number": 11,          // 币种数量
    "window_size": 50,          // 历史窗口大小
    "start_date": "2015/06/01", // 数据开始日期
    "end_date": "2017/06/01",   // 数据结束日期
    "test_portion": 0.15        // 测试集比例
  },
  "training": {
    "steps": 30000,             // 训练步数
    "learning_rate": 0.00028,   // 学习率
    "batch_size": 109,          // 批次大小
    "loss_function": "loss_function5",  // 损失函数
    "training_method": "Adam"   // 优化器
  },
  "trading": {
    "trading_consumption": 0.0025  // 交易佣金率 (0.25%)
  }
}
```

## 💡 完整工作流示例

```bash
# 1. 激活环境
source .venv/bin/activate  # 如果使用 uv

# 2. 首次运行：下载数据
python main.py --mode=download_data

# 3. 生成 3 个不同的训练配置
python main.py --mode=generate --repeat=3

# 4. 训练模型 1
python main.py --mode=train --processes=1 --device=cuda

# 5. 回测模型
python main.py --mode=backtest --algo=1

# 6. 对比结果
python main.py --mode=plot --algos=crp,olmar,1,2,3 --labels="CRP,OLMAR,模型1,模型2,模型3"

# 7. 生成结果表格
python main.py --mode=table --algos=1,2,3 --format=csv
```



## 📊 训练监控

### 使用 TensorBoard
```bash
tensorboard --logdir=./train_package/ --port=6006
```
访问 http://localhost:6006 查看：
- 训练损失曲线
- 投资组合价值
- 测试集性能

### 查看日志
```bash
# 查看训练日志
tail -f ./train_package/1/programlog

# 查看训练摘要
cat ./train_package/train_summary.csv
```

## 🎓 下一步

- 阅读 [README.md](README.md) 了解项目详情
- 查看 [user_guide.md](user_guide.md) 学习高级用法
- 阅读 [migration_report.md](migration_report.md) 了解 PyTorch 迁移细节

## 📝 注意事项

1. **数据**：数据请放在与项目根目录相同的database文件夹下
2. **训练时间**：完整训练（80000 步）在 CPU 上约需 20-30 分钟


## 🤝 需要帮助？

如有问题，请：
1. 查看 [Issues](https://github.com/yourusername/PGPortfolio/issues)
2. 提交新的 Issue
3. 查看项目文档

祝训练愉快！🚀