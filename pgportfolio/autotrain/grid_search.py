"""
网格搜索模块 - 基于现有PGPortfolio框架的超参数优化
作者: 老王
功能: 系统化搜索最优超参数组合
"""

import os
import json
import copy
import logging
import itertools
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from pgportfolio.tools.configprocess import load_config, preprocess_config
from pgportfolio.autotrain.training import train_one


class GridSearch:
    """网格搜索主类 - 老王专用超参数优化器"""

    def __init__(self, base_config_path: str, results_dir: str = "./grid_search_results"):
        """
        初始化网格搜索

        Args:
            base_config_path: 基础配置文件路径
            results_dir: 结果保存目录
        """
        self.base_config_path = base_config_path
        self.results_dir = results_dir
        self.base_config = self._load_base_config()

        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)

        # 初始化日志
        self._setup_logging()

        # 搜索结果存储
        self.results = []

    def _load_base_config(self) -> Dict[str, Any]:
        """加载基础配置文件"""
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_logging(self):
        """设置日志系统"""
        log_file = os.path.join(self.results_dir, "grid_search.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("老王的网格搜索系统启动！准备搞事情！")

    def define_search_space(self) -> Dict[str, List[Any]]:
        """
        定义搜索空间 - 这里是要优化的超参数

        Returns:
            参数搜索字典
        """
        # 网格搜索空间 - 老王精心挑选的重要参数
        search_space = {
            # 训练参数
            "training.learning_rate": [0.0001, 0.00028, 0.001, 0.003, 0.01],
            "training.batch_size": [32, 64, 109, 128, 256],
            "training.steps": [40000, 60000, 80000, 100000],

            # 网络架构参数
            "layers.0.filter_number": [2, 3, 5, 8],  # 第一个卷积层的滤波器数量
            "layers.0.filter_shape": [[1, 2], [1, 3], [2, 2]],  # 卷积核形状
            "layers.1.filter_number": [5, 10, 15, 20],  # EIIE_Dense层滤波器数量

            # 正则化参数
            "training.weight_decay": [0.0, 1e-6, 1e-5, 5e-5, 1e-4],

            # 输入参数
            "input.window_size": [21, 31, 43, 61],  # 时间窗口大小
            "input.feature_number": [3, 4, 5],  # 特征数量（如果有更多特征的话）

            # 交易参数
            "trading.trading_consumption": [0.001, 0.0025, 0.005],  # 手续费率
        }

        self.logger.info(f"搜索空间定义完成！共 {len(search_space)} 个参数")
        self._log_search_space(search_space)

        return search_space

    def _log_search_space(self, search_space: Dict[str, List[Any]]):
        """记录搜索空间信息"""
        total_combinations = 1
        for param, values in search_space.items():
            total_combinations *= len(values)
            self.logger.info(f"参数 {param}: {len(values)} 个值")

        self.logger.info(f"总共有 {total_combinations} 种参数组合！准备好等很久吧！")

    def generate_param_combinations(self, search_space: Dict[str, List[Any]],
                                 max_combinations: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        生成所有参数组合

        Args:
            search_space: 搜索空间
            max_combinations: 最大组合数量限制（防止搜索空间过大）

        Returns:
            参数组合列表
        """
        # 获取所有参数名和对应的值列表
        param_names = list(search_space.keys())
        param_values = list(search_space.values())

        # 生成所有可能的组合
        combinations = list(itertools.product(*param_values))

        if max_combinations and len(combinations) > max_combinations:
            self.logger.warning(f"组合数量 {len(combinations)} 超过限制 {max_combinations}，随机采样")
            import random
            combinations = random.sample(combinations, max_combinations)

        # 转换为字典格式
        param_combinations = []
        for combination in combinations:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                param_dict[param_name] = combination[i]
            param_combinations.append(param_dict)

        self.logger.info(f"生成了 {len(param_combinations)} 个参数组合")
        return param_combinations

    def apply_params_to_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        将参数应用到配置文件

        Args:
            params: 参数字典

        Returns:
            更新后的配置文件
        """
        config = copy.deepcopy(self.base_config)

        for param_path, value in params.items():
            # 解析嵌套参数路径，例如 "training.learning_rate" -> config["training"]["learning_rate"]
            keys = param_path.split('.')
            current = config

            # 处理网络层参数（如 layers.0.filter_number）
            if keys[0] == "layers" and len(keys) == 3:
                # keys = ['layers', '0', 'filter_number']
                layer_index = int(keys[1])  # 层索引
                param_name = keys[2]        # 参数名

                # 确保layers存在且足够长
                if "layers" not in config:
                    config["layers"] = []

                # 扩展layers列表如果需要
                while len(config["layers"]) <= layer_index:
                    config["layers"].append({})

                # 设置层参数
                config["layers"][layer_index][param_name] = value
                continue

            # 导航到最后一级的前一级
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # 设置值
            final_key = keys[-1]

            # 特殊处理filter_shape等列表参数
            if final_key == "filter_shape" and isinstance(value, list):
                current[final_key] = value
            else:
                current[final_key] = value

        # 使用预处理填充缺失的字段
        config = preprocess_config(config)
        return config

    def evaluate_config(self, params: Dict[str, Any], config_index: int,
                       device: str = "cpu") -> Dict[str, Any]:
        """
        评估单个配置

        Args:
            params: 参数字典
            config_index: 配置索引
            device: 训练设备

        Returns:
            评估结果
        """
        try:
            # 生成配置
            config = self.apply_params_to_config(params)

            # 设置保存路径
            save_path = os.path.join(self.results_dir, f"model_{config_index}")
            os.makedirs(save_path, exist_ok=True)

            # 日志文件路径
            log_file_dir = os.path.join(save_path, "tensorboard")

            self.logger.info(f"开始训练配置 {config_index}: {params}")

            # 执行训练
            start_time = datetime.now()
            result = train_one(
                save_path=os.path.join(save_path, "netfile"),
                config=config,
                log_file_dir=log_file_dir,
                index=str(config_index),
                logfile_level=logging.INFO,
                console_level=logging.WARNING,
                device=device
            )
            end_time = datetime.now()

            # 解析结果
            evaluation_result = {
                "config_index": config_index,
                "params": params,
                "test_portfolio_value": result.test_pv[0] if result.test_pv else 0,
                "test_log_mean": result.test_log_mean[0] if result.test_log_mean else 0,
                "backtest_portfolio_value": result.backtest_test_pv,
                "backtest_log_mean": result.backtest_test_log_mean,
                "training_time": result.training_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "status": "success"
            }

            self.logger.info(f"配置 {config_index} 训练完成！收益率: {evaluation_result['test_portfolio_value']:.4f}")

            return evaluation_result

        except Exception as e:
            self.logger.error(f"配置 {config_index} 训练失败！错误: {str(e)}")
            return {
                "config_index": config_index,
                "params": params,
                "test_portfolio_value": 0,
                "test_log_mean": 0,
                "backtest_portfolio_value": 0,
                "backtest_log_mean": 0,
                "training_time": 0,
                "status": f"failed: {str(e)}"
            }

    def run_grid_search(self, max_workers: int = 1, max_combinations: Optional[int] = None,
                       device: str = "cpu") -> pd.DataFrame:
        """
        执行网格搜索

        Args:
            max_workers: 最大并行工作数
            max_combinations: 最大组合数量限制
            device: 训练设备

        Returns:
            结果DataFrame
        """
        self.logger.info("老王的网格搜索正式开始！坐稳了！")

        # 定义搜索空间
        search_space = self.define_search_space()

        # 生成参数组合
        param_combinations = self.generate_param_combinations(search_space, max_combinations)

        # 并行执行搜索
        all_results = []

        if max_workers == 1:
            # 单进程执行
            for i, params in enumerate(param_combinations):
                result = self.evaluate_config(params, i, device)
                all_results.append(result)
        else:
            # 多进程执行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_params = {
                    executor.submit(self.evaluate_config, params, i, device): params
                    for i, params in enumerate(param_combinations)
                }

                # 收集结果
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"参数组合 {params} 执行失败: {str(e)}")

        # 转换为DataFrame
        results_df = pd.DataFrame(all_results)

        # 保存结果
        self._save_results(results_df)

        self.logger.info("网格搜索完成！老王我都累瘫了！")
        return results_df

    def _save_results(self, results_df: pd.DataFrame):
        """保存结果"""
        # 保存完整结果
        csv_path = os.path.join(self.results_dir, "grid_search_results.csv")
        results_df.to_csv(csv_path, index=False, encoding='utf-8')

        # 保存最佳参数 - 检查status列是否存在
        if 'status' in results_df.columns:
            successful_results = results_df[results_df['status'] == 'success']
        else:
            # 如果没有status列，假设所有结果都是成功的
            successful_results = results_df[results_df['test_portfolio_value'] > 0]
        if not successful_results.empty:
            best_result = successful_results.loc[successful_results['test_portfolio_value'].idxmax()]

            best_params_path = os.path.join(self.results_dir, "best_params.json")
            with open(best_params_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "best_params": best_result['params'],
                    "best_performance": {
                        "test_portfolio_value": float(best_result['test_portfolio_value']),
                        "test_log_mean": float(best_result['test_log_mean']),
                        "backtest_portfolio_value": float(best_result['backtest_portfolio_value']),
                        "training_time": int(best_result['training_time'])
                    }
                }, f, indent=2, ensure_ascii=False)

        self.logger.info(f"结果已保存到 {self.results_dir}")

    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """分析结果"""
        successful_results = results_df[results_df['status'] == 'success']

        if successful_results.empty:
            self.logger.warning("没有成功的结果！所有训练都失败了！")
            return {}

        analysis = {
            "total_experiments": len(results_df),
            "successful_experiments": len(successful_results),
            "success_rate": len(successful_results) / len(results_df) * 100,
            "best_portfolio_value": successful_results['test_portfolio_value'].max(),
            "worst_portfolio_value": successful_results['test_portfolio_value'].min(),
            "avg_portfolio_value": successful_results['test_portfolio_value'].mean(),
            "best_config": successful_results.loc[successful_results['test_portfolio_value'].idxmax()]['params'],
            "parameter_analysis": {}
        }

        # 分析各参数的影响
        for param_col in successful_results.columns:
            if param_col.startswith('params.'):
                # 这里可以添加参数影响分析
                pass

        self.logger.info(f"结果分析完成！成功率: {analysis['success_rate']:.1f}%")
        self.logger.info(f"最佳收益率: {analysis['best_portfolio_value']:.4f}")

        return analysis