# 网格搜索系统使用指南

## 🔥 功能概述

基于现有的训练框架，不破坏任何原有代码结构，帮你系统化地找到最优超参数组合。

## 📦 文件结构

```
PGPortfolio/
├── pgportfolio/autotrain/grid_search.py    # 核心网格搜索类
├── pgportfolio/resultprocess/grid_analysis.py  # 结果分析模块
├── run_grid_search.py                      # 完整网格搜索脚本
├── quick_grid_search.py                    # 快速网格搜索脚本
└── GRID_SEARCH_README.md                   # 本说明文档
```

## 🚀 快速开始

### 1. 快速测试（推荐先运行）

```bash
# 测试网格搜索是否正常工作
python quick_grid_search.py --test

# 预计运行时间: 5-10分钟
# 参数组合: 4个
# 输出: ./quick_grid_results/
```

### 2. 小规模搜索

```bash
# 小规模参数搜索
python quick_grid_search.py --small

# 预计运行时间: 30-60分钟
# 参数组合: 10个
# 输出: ./quick_grid_results/
```

### 3. 中等规模搜索

```bash
# 中等规模参数搜索
python quick_grid_search.py --medium --workers 2

# 预计运行时间: 2-4小时
# 参数组合: 50个
# 输出: ./quick_grid_results/
```

### 4. 完整网格搜索

```bash
# 完整的网格搜索（功能最全）
python run_grid_search.py --max_workers 4 --max_combinations 100

# 预计运行时间: 4-8小时
# 参数组合: 最多100个
# 输出: ./grid_search_results/
```

## 🎯 搜索的参数

### 训练参数
- **learning_rate**: [0.0001, 0.00028, 0.001, 0.003, 0.01] - 学习率
- **batch_size**: [32, 64, 109, 128, 256] - 批次大小
- **steps**: [40000, 60000, 80000, 100000] - 训练步数
- **weight_decay**: [0.0, 1e-6, 1e-5, 5e-5, 1e-4] - L2正则化

### 网络架构参数
- **layers.0.filter_number**: [2, 3, 5, 8] - 第一个卷积层滤波器数
- **layers.0.filter_shape**: [[1,2], [1,3], [2,2]] - 卷积核形状
- **layers.1.filter_number**: [5, 10, 15, 20] - EIIE_Dense层滤波器数

### 输入参数
- **input.window_size**: [21, 31, 43, 61] - 时间窗口大小
- **input.feature_number**: [3, 4, 5] - 特征数量

### 交易参数
- **trading.trading_consumption**: [0.001, 0.0025, 0.005] - 手续费率

## 📊 结果输出

搜索完成后会生成以下文件：

```
grid_search_results/
├── grid_search_results.csv        # 详细结果数据
├── best_params.json               # 最佳参数配置
├── grid_search.log               # 搜索日志
├── plots/                        # 可视化图表
│   ├── results_overview.png      # 结果概览图
│   └── parameter_importance.png  # 参数重要性分析
└── analysis/                     # 深度分析报告
    ├── analysis_report.json      # JSON格式分析报告
    └── analysis_report.txt       # 可读性强的文本报告
```

## 📈 结果分析

### 关键指标
- **test_portfolio_value**: 测试集投资组合价值（主要优化目标）
- **test_log_mean**: 对数平均收益率
- **backtest_portfolio_value**: 回测收益率
- **training_time**: 训练耗时

### 分析图表
1. **收益率分布直方图** - 了解性能分布
2. **训练时间 vs 收益率散点图** - 评估效率
3. **参数重要性分析** - 找出关键参数
4. **学习率 vs 批次大小热力图** - 参数组合效果
5. **窗口大小性能对比** - 时序窗口影响

### 智能建议
系统会自动生成优化建议：
- 🎯 参数调整建议（基于相关性分析）
- ⚠️ 风险警告（过拟合、训练稳定性等）
- 📋 下一步行动计划

## ⚙️ 高级配置

### 自定义搜索空间

如果需要搜索其他参数，可以修改 `grid_search.py` 中的 `define_search_space()` 方法：

```python
def define_search_space(self) -> Dict[str, List[Any]]:
    search_space = {
        "training.learning_rate": [0.0001, 0.001, 0.01],  # 自定义学习率
        "custom_param": [value1, value2, value3],          # 自定义参数
    }
    return search_space
```

### 并行训练

```bash
# 使用4个CPU核心并行训练
python run_grid_search.py --max_workers 4

# 使用GPU训练（如果支持）
python run_grid_search.py --device cuda
```

### 搜索限制

```bash
# 限制最大组合数量（防止搜索空间过大）
python run_grid_search.py --max_combinations 50
```

## 🐛 常见问题

### Q: 训练失败率高怎么办？
A: 检查以下方面：
1. 数据质量是否正常
2. 内存是否充足
3. 某些参数组合是否导致数值不稳定

### Q: 搜索速度太慢？
A: 可以：
1. 减少 `max_combinations` 数量
2. 使用 `--workers` 增加并行进程
3. 使用快速测试模式验证功能

### Q: 结果不理想？
A: 尝试：
1. 扩大搜索空间
2. 检查参数范围是否合理
3. 参考参数重要性分析调整搜索重点

### Q: 如何应用到实际训练？
A: 使用 `best_params.json` 中的配置：
```python
import json
with open('grid_search_results/best_params.json') as f:
    best_config = json.load(f)
    # 使用 best_config["best_params"] 进行训练
```

## 🔧 技术细节

### 兼容性保证
- ✅ 完全兼容现有训练框架
- ✅ 不修改任何核心代码
- ✅ 支持所有原有参数
- ✅ 结果格式一致

### 搜索策略
- **网格搜索**: 系统化遍历所有参数组合
- **并行执行**: 多进程训练加速搜索
- **智能截断**: 支持最大组合数限制
- **容错处理**: 失败实验不影响整体搜索

### 分析方法
- **统计分析**: 基础统计指标计算
- **相关性分析**: 参数与性能相关性
- **可视化分析**: 多维度图表展示
- **智能推荐**: 基于结果的优化建议

## 📞 建议

1. **先跑测试模式** - 确保系统正常工作
2. **循序渐进** - 从小规模搜索开始
3. **关注成功率** - 失败率高说明参数范围不合理
4. **重视相关性** - 相关性高的参数值得重点优化
5. **验证结果** - 最好用独立数据验证最佳配置

---

**🔥 老王的网格搜索系统 - 让超参数优化变简单！**

有问题？检查日志文件 `grid_search.log`，老王把所有信息都记录在里面了！