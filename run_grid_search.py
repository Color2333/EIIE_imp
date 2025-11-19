#!/usr/bin/env python3
"""
网格搜索执行脚本 - 老王专用
使用方法: python run_grid_search.py

这个脚本会：
1. 基于现有配置进行网格搜索
2. 自动训练和评估不同参数组合
3. 找出最优超参数配置
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgportfolio.autotrain.grid_search import GridSearch


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="老王的网格搜索工具")

    parser.add_argument("--config", default="./pgportfolio/net_config.json",
                       help="基础配置文件路径 (默认: ./pgportfolio/net_config.json)")

    parser.add_argument("--results_dir", default="./grid_search_results",
                       help="结果保存目录 (默认: ./grid_search_results)")

    parser.add_argument("--max_workers", type=int, default=1,
                       help="最大并行进程数 (默认: 1)")

    parser.add_argument("--max_combinations", type=int, default=None,
                       help="最大参数组合数量限制 (默认: 无限制)")

    parser.add_argument("--device", default="cpu",
                       help="训练设备 (默认: cpu)")

    parser.add_argument("--quick_test", action="store_true",
                       help="快速测试模式 - 使用少量参数组合")

    return parser.parse_args()


def run_quick_test():
    """快速测试模式 - 用少量参数组合验证网格搜索是否工作"""
    print("🔥 老王的快速测试模式启动！")

    # 创建测试用的搜索空间
    search_space = {
        "training.learning_rate": [0.00028, 0.001],
        "training.batch_size": [64, 109],
        "input.window_size": [31, 43],
    }

    grid_search = GridSearch("./pgportfolio/net_config.json")

    # 手动生成少量组合进行测试
    test_combinations = [
        {"training.learning_rate": 0.00028, "training.batch_size": 64, "input.window_size": 31},
        {"training.learning_rate": 0.001, "training.batch_size": 109, "input.window_size": 43},
    ]

    print(f"测试模式：运行 {len(test_combinations)} 个参数组合")

    results = []
    for i, params in enumerate(test_combinations):
        print(f"\n--- 测试配置 {i+1}: {params} ---")
        result = grid_search.evaluate_config(params, i, device="cpu")
        results.append(result)
        print(f"收益率: {result['test_portfolio_value']:.4f}")

    # 保存测试结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("./quick_test_results.csv", index=False)
    print(f"\n测试完成！结果已保存到 ./quick_test_results.csv")

    return results_df


def visualize_results(results_df, results_dir):
    """可视化结果"""
    print("\n📊 老王开始生成结果图表...")

    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 创建图表目录
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. 收益率分布图
    successful_results = results_df[results_df['status'] == 'success']

    if not successful_results.empty:
        plt.figure(figsize=(12, 8))

        # 子图1: 收益率分布
        plt.subplot(2, 2, 1)
        plt.hist(successful_results['test_portfolio_value'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('投资组合价值')
        plt.ylabel('频次')
        plt.title('收益率分布直方图')
        plt.grid(True, alpha=0.3)

        # 子图2: 训练时间 vs 收益率
        plt.subplot(2, 2, 2)
        plt.scatter(successful_results['training_time'], successful_results['test_portfolio_value'],
                   alpha=0.6, s=50)
        plt.xlabel('训练时间 (秒)')
        plt.ylabel('投资组合价值')
        plt.title('训练时间 vs 收益率')
        plt.grid(True, alpha=0.3)

        # 子图3: 学习率影响分析
        if 'params.training.learning_rate' in successful_results.columns:
            lr_results = successful_results.groupby('params.training.learning_rate')['test_portfolio_value'].mean()
            plt.subplot(2, 2, 3)
            lr_results.plot(kind='bar', alpha=0.7)
            plt.xlabel('学习率')
            plt.ylabel('平均收益率')
            plt.title('不同学习率的性能')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        # 子图4: 批次大小影响分析
        if 'params.training.batch_size' in successful_results.columns:
            batch_results = successful_results.groupby('params.training.batch_size')['test_portfolio_value'].mean()
            plt.subplot(2, 2, 4)
            batch_results.plot(kind='bar', alpha=0.7, color='orange')
            plt.xlabel('批次大小')
            plt.ylabel('平均收益率')
            plt.title('不同批次大小的性能')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "results_overview.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"结果图表已保存到: {plot_path}")

    # 生成参数重要性分析（如果有足够的数据）
    if len(successful_results) >= 5:
        try:
            create_parameter_importance_plot(successful_results, plots_dir)
        except Exception as e:
            print(f"参数重要性分析失败: {str(e)}")


def create_parameter_importance_plot(successful_results, plots_dir):
    """创建参数重要性分析图"""
    print("🔍 分析参数重要性...")

    # 提取数值型参数
    numeric_params = []
    param_correlations = {}

    for col in successful_results.columns:
        if col.startswith('params.') and successful_results[col].dtype in ['int64', 'float64']:
            param_name = col.replace('params.', '')
            correlation = successful_results[col].corr(successful_results['test_portfolio_value'])
            if not pd.isna(correlation):
                numeric_params.append(param_name)
                param_correlations[param_name] = correlation

    if param_correlations:
        # 创建相关性热力图
        plt.figure(figsize=(10, 6))

        params = list(param_correlations.keys())
        correlations = list(param_correlations.values())

        colors = ['red' if c < 0 else 'green' for c in correlations]
        bars = plt.bar(range(len(params)), correlations, color=colors, alpha=0.7)

        plt.xlabel('参数')
        plt.ylabel('与收益率的相关性')
        plt.title('参数重要性分析')
        plt.xticks(range(len(params)), params, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 if corr > 0 else bar.get_height() - 0.001,
                    f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()

        importance_path = os.path.join(plots_dir, "parameter_importance.png")
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"参数重要性图已保存到: {importance_path}")


def main():
    """主函数"""
    print("🚀 老王的超参数网格搜索系统启动！")
    print("=" * 50)

    args = parse_arguments()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)

    # 快速测试模式
    if args.quick_test:
        print("🧪 运行快速测试模式...")
        try:
            results_df = run_quick_test()
            print("\n✅ 快速测试完成！")
            print(f"平均收益率: {results_df['test_portfolio_value'].mean():.4f}")
            print("如果测试正常，可以运行完整网格搜索：")
            print("python run_grid_search.py --max_workers 4")
        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
        return

    # 完整网格搜索
    try:
        print(f"📋 配置文件: {args.config}")
        print(f"💾 结果目录: {args.results_dir}")
        print(f"🔧 并行进程数: {args.max_workers}")
        print(f"💻 训练设备: {args.device}")

        # 创建网格搜索对象
        grid_search = GridSearch(
            base_config_path=args.config,
            results_dir=args.results_dir
        )

        # 执行网格搜索
        results_df = grid_search.run_grid_search(
            max_workers=args.max_workers,
            max_combinations=args.max_combinations,
            device=args.device
        )

        # 分析结果
        analysis = grid_search.analyze_results(results_df)

        # 打印摘要
        print("\n" + "=" * 50)
        print("📈 网格搜索结果摘要:")
        print(f"总实验数: {analysis.get('total_experiments', 0)}")
        print(f"成功实验数: {analysis.get('successful_experiments', 0)}")
        print(f"成功率: {analysis.get('success_rate', 0):.1f}%")
        print(f"最佳收益率: {analysis.get('best_portfolio_value', 0):.4f}")
        print(f"平均收益率: {analysis.get('avg_portfolio_value', 0):.4f}")

        # 生成可视化
        visualize_results(results_df, args.results_dir)

        print("\n🎉 网格搜索完成！")
        print(f"📁 详细结果保存在: {args.results_dir}")
        print("🔥 老王的超参数优化大功告成！")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断搜索")
    except Exception as e:
        print(f"\n❌ 网格搜索失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()