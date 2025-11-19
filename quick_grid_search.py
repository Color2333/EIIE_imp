#!/usr/bin/env python3
"""
快速网格搜索启动脚本 - 老王专用简化版
适合快速测试和验证网格搜索功能

使用方法:
1. 快速测试: python quick_grid_search.py --test
2. 小规模搜索: python quick_grid_search.py --small
3. 中等规模: python quick_grid_search.py --medium
"""

import sys
import os
import argparse
from pgportfolio.autotrain.grid_search import GridSearch


def create_test_search_space():
    """创建测试用的搜索空间"""
    return {
        "training.learning_rate": [0.00028, 0.001],
        "training.batch_size": [64, 109],
        "input.window_size": [31],
    }


def create_small_search_space():
    """创建小规模搜索空间"""
    return {
        "training.learning_rate": [0.0001, 0.00028, 0.001],
        "training.batch_size": [64, 109, 128],
        "input.window_size": [31, 43],
        "layers.0.filter_number": [2, 3, 5],
    }


def create_medium_search_space():
    """创建中等规模搜索空间"""
    return {
        "training.learning_rate": [0.0001, 0.00028, 0.001, 0.003],
        "training.batch_size": [64, 109, 128, 256],
        "training.steps": [40000, 60000, 80000],
        "input.window_size": [21, 31, 43],
        "layers.0.filter_number": [2, 3, 5, 8],
        "layers.1.filter_number": [5, 10, 15],
        "training.weight_decay": [0.0, 1e-6, 1e-5],
    }


def run_quick_search(search_space, results_dir, max_workers=1, max_combinations=None):
    """运行快速搜索"""
    print(f"🚀 启动老王的快速网格搜索！")
    print(f"搜索空间大小: {len(search_space)} 个参数")

    # 估算组合数量
    total_combinations = 1
    for param, values in search_space.items():
        total_combinations *= len(values)

    if max_combinations:
        total_combinations = min(total_combinations, max_combinations)

    print(f"预计运行 {total_combinations} 个组合")

    # 创建自定义网格搜索类
    class CustomGridSearch(GridSearch):
        def __init__(self, base_config_path, results_dir, search_space):
            super().__init__(base_config_path, results_dir)
            self.custom_search_space = search_space

        def define_search_space(self):
            return self.custom_search_space

    # 创建网格搜索对象
    grid_search = CustomGridSearch(
        base_config_path="./pgportfolio/net_config.json",
        results_dir=results_dir,
        search_space=search_space
    )

    # 执行搜索（使用单进程避免pickle问题）
    try:
        # 暂时强制使用单进程避免pickle问题
        print("🔧 使用单进程模式以避免序列化问题...")
        results_df = grid_search.run_grid_search(
            max_workers=1,  # 强制单进程
            max_combinations=max_combinations,
            device="cpu"
        )

        # 分析结果
        analysis = grid_search.analyze_results(results_df)

        # 打印结果摘要
        print("\n" + "="*50)
        print("📊 快速搜索结果:")
        print(f"总实验数: {analysis.get('total_experiments', 0)}")
        print(f"成功实验数: {analysis.get('successful_experiments', 0)}")
        print(f"成功率: {analysis.get('success_rate', 0):.1f}%")
        print(f"最佳收益率: {analysis.get('best_portfolio_value', 0):.4f}")
        print(f"平均收益率: {analysis.get('avg_portfolio_value', 0):.4f}")

        if 'best_config' in analysis:
            print(f"\n🏆 最佳配置:")
            print(f"收益率: {analysis['best_config']['portfolio_value']:.4f}")
            print(f"训练时间: {analysis['best_config']['training_time']}秒")
            print(f"参数: {analysis['best_config']['params']}")

        print(f"\n💾 详细结果保存在: {results_dir}")

        return results_df

    except Exception as e:
        print(f"❌ 搜索失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="老王的快速网格搜索工具")
    parser.add_argument("--test", action="store_true", help="运行测试模式 (2个组合)")
    parser.add_argument("--small", action="store_true", help="运行小规模搜索 (18个组合)")
    parser.add_argument("--medium", action="store_true", help="运行中等规模搜索 (288个组合)")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数")
    parser.add_argument("--results_dir", default="./quick_grid_results", help="结果保存目录")

    args = parser.parse_args()

    # 检查配置文件
    if not os.path.exists("./pgportfolio/net_config.json"):
        print("❌ 找不到配置文件 ./pgportfolio/net_config.json")
        print("请确保在正确的项目目录中运行此脚本")
        sys.exit(1)

    # 选择搜索空间
    if args.test:
        search_space = create_test_search_space()
        max_combinations = 4  # 限制测试数量
        print("🧪 运行测试模式...")
    elif args.small:
        search_space = create_small_search_space()
        max_combinations = 10  # 限制小规模搜索数量
        print("🔍 运行小规模搜索...")
    elif args.medium:
        search_space = create_medium_search_space()
        max_combinations = 50  # 限制中等规模搜索数量
        print("🔎 运行中等规模搜索...")
    else:
        print("❌ 请选择搜索模式: --test, --small, 或 --medium")
        sys.exit(1)

    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)

    # 运行搜索（暂时忽略workers参数，使用单进程）
    results_df = run_quick_search(
        search_space=search_space,
        results_dir=args.results_dir,
        max_workers=1,  # 暂时强制单进程
        max_combinations=max_combinations
    )

    if results_df is not None:
        # 尝试生成简单的分析
        try:
            from pgportfolio.resultprocess.grid_analysis import GridSearchAnalyzer
            analyzer = GridSearchAnalyzer(
                results_path=os.path.join(args.results_dir, "grid_search_results.csv"),
                output_dir=os.path.join(args.results_dir, "analysis")
            )
            analyzer.generate_report()
            print("📈 已生成详细分析报告")
        except Exception as e:
            print(f"⚠️ 分析报告生成失败: {str(e)}")

        print("\n🎉 快速网格搜索完成！")
        print("可以查看详细结果和可视化图表")
    else:
        print("\n❌ 搜索失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()