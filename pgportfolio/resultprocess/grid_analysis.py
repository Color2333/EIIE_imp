"""
网格搜索结果分析模块 - 老王专用分析工具
功能: 深度分析网格搜索结果，提供洞察和建议
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import json
import os
from datetime import datetime


class GridSearchAnalyzer:
    """网格搜索结果分析器 - 老王的专业分析工具"""

    def __init__(self, results_path: str, output_dir: str = "./grid_analysis"):
        """
        初始化分析器

        Args:
            results_path: 网格搜索结果CSV文件路径
            output_dir: 分析结果输出目录
        """
        self.results_path = results_path
        self.output_dir = output_dir
        self.results_df = self._load_results()

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

    def _load_results(self) -> pd.DataFrame:
        """加载网格搜索结果"""
        try:
            df = pd.read_csv(self.results_path)
            print(f"📊 成功加载 {len(df)} 条网格搜索结果")
            return df
        except Exception as e:
            raise ValueError(f"加载结果文件失败: {str(e)}")

    def basic_statistics(self) -> Dict[str, Any]:
        """基础统计分析"""
        successful = self.results_df[self.results_df['status'] == 'success']

        if successful.empty:
            return {"error": "没有成功的结果"}

        stats = {
            "total_experiments": len(self.results_df),
            "successful_experiments": len(successful),
            "success_rate": len(successful) / len(self.results_df) * 100,
            "portfolio_stats": {
                "best": successful['test_portfolio_value'].max(),
                "worst": successful['test_portfolio_value'].min(),
                "mean": successful['test_portfolio_value'].mean(),
                "median": successful['test_portfolio_value'].median(),
                "std": successful['test_portfolio_value'].std(),
                "q25": successful['test_portfolio_value'].quantile(0.25),
                "q75": successful['test_portfolio_value'].quantile(0.75),
            },
            "training_time_stats": {
                "mean": successful['training_time'].mean(),
                "median": successful['training_time'].median(),
                "total": successful['training_time'].sum(),
            }
        }

        # 找出最佳配置
        best_idx = successful['test_portfolio_value'].idxmax()
        best_config = successful.loc[best_idx]

        stats["best_config"] = {
            "portfolio_value": float(best_config['test_portfolio_value']),
            "log_mean": float(best_config['test_log_mean']),
            "training_time": int(best_config['training_time']),
            "config_index": int(best_config['config_index']),
            "params": best_config['params'] if 'params' in best_config else {}
        }

        return stats

    def parameter_importance_analysis(self) -> Dict[str, float]:
        """参数重要性分析"""
        successful = self.results_df[self.results_df['status'] == 'success']

        if successful.empty:
            return {}

        # 提取数值型参数
        numeric_params = []
        param_importance = {}

        for col in successful.columns:
            if col.startswith('params.') and successful[col].dtype in ['int64', 'float64']:
                # 计算与收益率的相关性
                correlation = successful[col].corr(successful['test_portfolio_value'])
                if not pd.isna(correlation):
                    param_name = col.replace('params.', '')
                    param_importance[param_name] = correlation

        # 按绝对相关性排序
        param_importance = dict(sorted(param_importance.items(),
                                      key=lambda x: abs(x[1]), reverse=True))

        return param_importance

    def categorical_parameter_analysis(self) -> Dict[str, Dict[str, float]]:
        """分类参数分析"""
        successful = self.results_df[self.results_df['status'] == 'success']

        if successful.empty:
            return {}

        categorical_analysis = {}

        for col in successful.columns:
            if col.startswith('params.') and successful[col].dtype == 'object':
                param_name = col.replace('params.', '')

                # 计算每个类别的平均性能
                category_performance = successful.groupby(col)['test_portfolio_value'].agg(['mean', 'count', 'std'])

                # 只保留样本数足够的类别
                valid_categories = category_performance[category_performance['count'] >= 2]

                if not valid_categories.empty:
                    categorical_analysis[param_name] = {}
                    for category, stats in valid_categories.iterrows():
                        categorical_analysis[param_name][str(category)] = {
                            'mean_performance': float(stats['mean']),
                            'sample_count': int(stats['count']),
                            'std': float(stats['std']) if not pd.isna(stats['std']) else 0
                        }

        return categorical_analysis

    def create_comprehensive_plots(self):
        """创建综合分析图表"""
        successful = self.results_df[self.results_df['status'] == 'success']

        if successful.empty:
            print("❌ 没有成功的结果，无法生成图表")
            return

        print("📈 生成综合分析图表...")

        # 创建大图
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. 收益率分布 (左上角，占用2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.hist(successful['test_portfolio_value'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.set_xlabel('投资组合价值', fontsize=12)
        ax1.set_ylabel('频次', fontsize=12)
        ax1.set_title('收益率分布直方图', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(successful['test_portfolio_value'].mean(), color='red', linestyle='--',
                   label=f'均值: {successful["test_portfolio_value"].mean():.4f}')
        ax1.legend()

        # 2. 训练时间 vs 收益率散点图 (右上角，占用2x2)
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        scatter = ax2.scatter(successful['training_time'], successful['test_portfolio_value'],
                            alpha=0.6, s=50, c=successful['test_portfolio_value'], cmap='viridis')
        ax2.set_xlabel('训练时间 (秒)', fontsize=12)
        ax2.set_ylabel('投资组合价值', fontsize=12)
        ax2.set_title('训练时间 vs 收益率', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='投资组合价值')

        # 3. 参数重要性分析 (左下角，占用2x2)
        ax3 = fig.add_subplot(gs[2:4, 0:2])
        param_importance = self.parameter_importance_analysis()

        if param_importance:
            params = list(param_importance.keys())[:10]  # 只显示前10个
            correlations = [param_importance[p] for p in params]
            colors = ['red' if c < 0 else 'green' for c in correlations]

            bars = ax3.barh(range(len(params)), correlations, color=colors, alpha=0.7)
            ax3.set_yticks(range(len(params)))
            ax3.set_yticklabels(params)
            ax3.set_xlabel('与收益率的相关性', fontsize=12)
            ax3.set_title('参数重要性分析', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # 添加数值标签
            for i, (bar, corr) in enumerate(zip(bars, correlations)):
                ax3.text(bar.get_width() + (0.001 if corr > 0 else -0.001),
                        bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', ha='left' if corr > 0 else 'right',
                        va='center', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, '没有足够的数值参数进行分析', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('参数重要性分析', fontsize=14, fontweight='bold')

        # 4. 学习率和批次大小的热力图 (右下角)
        ax4 = fig.add_subplot(gs[2, 2])
        lr_col = 'params.training.learning_rate'
        batch_col = 'params.training.batch_size'

        if lr_col in successful.columns and batch_col in successful.columns:
            pivot_table = successful.pivot_table(
                values='test_portfolio_value',
                index=lr_col,
                columns=batch_col,
                aggfunc='mean'
            )

            im = ax4.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
            ax4.set_xticks(range(len(pivot_table.columns)))
            ax4.set_xticklabels(pivot_table.columns)
            ax4.set_yticks(range(len(pivot_table.index)))
            ax4.set_yticklabels(pivot_table.index)
            ax4.set_xlabel('批次大小', fontsize=11)
            ax4.set_ylabel('学习率', fontsize=11)
            ax4.set_title('学习率 vs 批次大小性能热力图', fontsize=12, fontweight='bold')

            # 添加数值标签
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    text = ax4.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')

            plt.colorbar(im, ax=ax4, label='投资组合价值')
        else:
            ax4.text(0.5, 0.5, '学习率或批次大小数据不足', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=11)
            ax4.set_title('学习率 vs 批次大小分析', fontsize=12, fontweight='bold')

        # 5. 窗口大小分析
        ax5 = fig.add_subplot(gs[2, 3])
        window_col = 'params.input.window_size'

        if window_col in successful.columns:
            window_stats = successful.groupby(window_col)['test_portfolio_value'].agg(['mean', 'std'])
            x_pos = np.arange(len(window_stats))

            bars = ax5.bar(x_pos, window_stats['mean'], yerr=window_stats['std'],
                          capsize=5, alpha=0.7, color='lightcoral')
            ax5.set_xlabel('窗口大小', fontsize=11)
            ax5.set_ylabel('平均收益率', fontsize=11)
            ax5.set_title('不同窗口大小的性能', fontsize=12, fontweight='bold')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(window_stats.index)
            ax5.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, mean in zip(bars, window_stats['mean']):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{mean:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, '窗口大小数据不足', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=11)
            ax5.set_title('窗口大小分析', fontsize=12, fontweight='bold')

        # 6. 性能排名和改进建议 (最后一个位置)
        ax6 = fig.add_subplot(gs[3, 2:])
        ax6.axis('off')

        # 生成性能排名
        top_configs = successful.nlargest(5, 'test_portfolio_value')[['config_index', 'test_portfolio_value', 'training_time']]

        text_content = "🏆 最佳5个配置:\n\n"
        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            text_content += f"{i}. 配置{int(row['config_index'])}: {row['test_portfolio_value']:.4f} (耗时{int(row['training_time'])}s)\n"

        # 添加参数重要性摘要
        param_importance = self.parameter_importance_analysis()
        if param_importance:
            text_content += "\n🔍 关键参数影响:\n"
            for param, corr in list(param_importance.items())[:3]:
                direction = "正相关" if corr > 0 else "负相关"
                text_content += f"• {param}: {direction} ({corr:.3f})\n"

        ax6.text(0.05, 0.95, text_content, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.suptitle('老王的网格搜索综合分析报告', fontsize=16, fontweight='bold', y=0.98)

        # 保存大图
        plot_path = os.path.join(self.output_dir, "comprehensive_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"综合分析图已保存到: {plot_path}")

    def generate_recommendations(self) -> Dict[str, Any]:
        """生成优化建议"""
        stats = self.basic_statistics()
        param_importance = self.parameter_importance_analysis()

        recommendations = {
            "overall_assessment": self._get_overall_assessment(stats),
            "parameter_recommendations": self._get_parameter_recommendations(param_importance),
            "next_steps": self._get_next_steps(),
            "risk_warnings": self._get_risk_warnings(stats)
        }

        return recommendations

    def _get_overall_assessment(self, stats: Dict[str, Any]) -> str:
        """获取整体评估"""
        if "error" in stats:
            return "无法生成评估 - 没有成功的结果"

        success_rate = stats["success_rate"]
        best_return = stats["portfolio_stats"]["best"]
        avg_return = stats["portfolio_stats"]["mean"]

        if success_rate < 50:
            assessment = "❌ 训练稳定性差，成功率过低"
        elif success_rate < 80:
            assessment = "⚠️ 训练稳定性一般，需要调参"
        else:
            assessment = "✅ 训练稳定性良好"

        if best_return > 1.5:
            assessment += "\n🚀 发现了高性能配置，收益率优秀"
        elif best_return > 1.2:
            assessment += "\n👍 性能良好，有改进空间"
        else:
            assessment += "\n📉 整体性能偏低，需要大幅优化"

        return assessment

    def _get_parameter_recommendations(self, param_importance: Dict[str, float]) -> List[str]:
        """获取参数建议"""
        recommendations = []

        for param, correlation in list(param_importance.items())[:5]:
            if abs(correlation) < 0.1:
                continue  # 相关性太低的参数跳过

            if "learning_rate" in param:
                if correlation > 0:
                    recommendations.append(f"🎯 {param}: 正相关，建议适当增加学习率")
                else:
                    recommendations.append(f"🎯 {param}: 负相关，建议降低学习率")
            elif "batch_size" in param:
                if correlation > 0:
                    recommendations.append(f"🎯 {param}: 正相关，建议使用更大的批次大小")
                else:
                    recommendations.append(f"🎯 {param}: 负相关，建议使用较小的批次大小")
            elif "window_size" in param:
                if correlation > 0:
                    recommendations.append(f"🎯 {param}: 正相关，建议增加时间窗口")
                else:
                    recommendations.append(f"🎯 {param}: 负相关，建议缩短时间窗口")
            else:
                direction = "增加" if correlation > 0 else "减少"
                recommendations.append(f"🎯 {param}: {direction}该参数可能提升性能")

        return recommendations

    def _get_next_steps(self) -> List[str]:
        """获取下一步建议"""
        return [
            "🔄 在最佳参数附近进行精细化搜索",
            "🎲 尝试贝叶斯优化等更高级的搜索方法",
            "📊 收集更多训练数据进行验证",
            "🧪 尝试不同的网络架构优化",
            "⚡ 实施早停策略避免过拟合"
        ]

    def _get_risk_warnings(self, stats: Dict[str, Any]) -> List[str]:
        """获取风险警告"""
        warnings = []

        if "portfolio_stats" in stats:
            std_dev = stats["portfolio_stats"]["std"]
            if std_dev > 0.5:
                warnings.append("⚠️ 收益率标准差过大，存在过拟合风险")

            q25 = stats["portfolio_stats"]["q25"]
            if q25 < 1.0:
                warnings.append("⚠️ 25%的配置出现亏损，需要改进鲁棒性")

        if stats.get("success_rate", 100) < 70:
            warnings.append("⚠️ 训练成功率偏低，检查数据质量")

        return warnings

    def generate_report(self):
        """生成完整分析报告"""
        print("📝 生成详细分析报告...")

        # 基础统计
        stats = self.basic_statistics()

        # 参数分析
        param_importance = self.parameter_importance_analysis()
        categorical_analysis = self.categorical_parameter_analysis()

        # 生成建议
        recommendations = self.generate_recommendations()

        # 组装报告
        report = {
            "generation_time": datetime.now().isoformat(),
            "data_source": self.results_path,
            "basic_statistics": stats,
            "parameter_importance": param_importance,
            "categorical_analysis": categorical_analysis,
            "recommendations": recommendations
        }

        # 保存报告
        report_path = os.path.join(self.output_dir, "analysis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 生成可读性报告
        self._generate_text_report(report)

        print(f"✅ 分析报告已保存到: {self.output_dir}")

    def _generate_text_report(self, report: Dict[str, Any]):
        """生成可读性强的文本报告"""
        report_path = os.path.join(self.output_dir, "analysis_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("老王的网格搜索深度分析报告\n")
            f.write("=" * 60 + "\n\n")

            # 基础统计
            stats = report["basic_statistics"]
            if "error" not in stats:
                f.write("📊 基础统计:\n")
                f.write(f"总实验数: {stats['total_experiments']}\n")
                f.write(f"成功实验数: {stats['successful_experiments']}\n")
                f.write(f"成功率: {stats['success_rate']:.1f}%\n")
                f.write(f"最佳收益率: {stats['portfolio_stats']['best']:.4f}\n")
                f.write(f"平均收益率: {stats['portfolio_stats']['mean']:.4f}\n")
                f.write(f"收益率标准差: {stats['portfolio_stats']['std']:.4f}\n\n")

            # 参数重要性
            param_importance = report["parameter_importance"]
            if param_importance:
                f.write("🔍 参数重要性分析:\n")
                for param, corr in list(param_importance.items())[:10]:
                    direction = "正相关" if corr > 0 else "负相关"
                    f.write(f"  {param}: {direction} (相关性: {corr:.3f})\n")
                f.write("\n")

            # 建议
            recommendations = report["recommendations"]
            f.write("💡 优化建议:\n")
            f.write(f"整体评估: {recommendations['overall_assessment']}\n\n")

            if recommendations['parameter_recommendations']:
                f.write("参数建议:\n")
                for rec in recommendations['parameter_recommendations']:
                    f.write(f"  {rec}\n")
                f.write("\n")

            if recommendations['risk_warnings']:
                f.write("风险警告:\n")
                for warning in recommendations['risk_warnings']:
                    f.write(f"  {warning}\n")
                f.write("\n")

            f.write("下一步行动:\n")
            for step in recommendations['next_steps']:
                f.write(f"  {step}\n")

        print(f"📄 文本报告已保存到: {report_path}")