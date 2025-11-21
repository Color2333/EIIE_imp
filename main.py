from __future__ import absolute_import
import json
import logging
import os
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime

from pgportfolio.tools.configprocess import preprocess_config
from pgportfolio.tools.configprocess import load_config
from pgportfolio.tools.trade import save_test_data
from pgportfolio.tools.shortcut import execute_backtest
from pgportfolio.resultprocess import plot
from pgportfolio.marketdata.data_manager import BinanceDataManager


def build_parser():
    parser = ArgumentParser(
        description="PGPortfolio - 加密货币投资组合管理系统",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
模式说明:
  train              训练模型
  generate           生成训练数据包
  download_data      下载数据 (旧版)
  backtest           回测模型
  save_test_data     保存测试数据
  plot               绘制回测结果
  table              显示回测表格

数据管理命令:
  data-create        创建数据库
  data-status        检查数据状态
  data-download      下载单个币种数据
  data-download-all  下载所有币种数据
  data-complete      补全缺失数据
  data-verify        验证数据库完整性
  data-monitor       监控下载进度
        """
    )

    # 主要模式参数
    parser.add_argument("--mode", dest="mode",
                        help="运行模式: train, generate, download_data, backtest, save_test_data, plot, table",
                        metavar="MODE", default="train")
    parser.add_argument("--processes", dest="processes",
                        help="训练进程数量", default="1")
    parser.add_argument("--repeat", dest="repeat",
                        help="生成训练包的重复次数", default="1")
    parser.add_argument("--algo", help="算法名称或训练包索引", dest="algo")
    parser.add_argument("--algos", help="算法名称或训练包索引，用逗号分隔", dest="algos")
    parser.add_argument("--labels", dest="labels",
                        help="图表标题或表格头部显示的名称")
    parser.add_argument("--format", dest="format", default="raw",
                        help="表格输出格式")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="训练使用的设备")
    parser.add_argument("--folder", dest="folder", type=int,
                        help="加载配置的文件夹编号，如果不提供则从./pgportfolio/net_config加载")

    # 数据管理参数
    parser.add_argument("--database", "-d", default="DataNew.db",
                        help="数据库文件名 (默认: DataNew.db)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="跳过确认提示")
    parser.add_argument("--append", action="store_true",
                        help="追加模式，用于补全已有数据")

    return parser


def main():
    parser = build_parser()

    # 特殊处理数据管理命令
    if len(sys.argv) > 1 and sys.argv[1].startswith('data-'):
        handle_data_commands()
        return

    options = parser.parse_args()

    # 创建必要目录
    if not os.path.exists("./" + "train_package"):
        os.makedirs("./" + "train_package")
    if not os.path.exists("./" + "database"):
        os.makedirs("./" + "database")

    # 处理传统模式
    if options.mode == "train":
        import pgportfolio.autotrain.training
        if not options.algo:
            pgportfolio.autotrain.training.train_all(int(options.processes), options.device)
        else:
            for folder in options.folder:
                raise NotImplementedError()
    elif options.mode == "generate":
        import pgportfolio.autotrain.generate as generate
        logging.basicConfig(level=logging.INFO)
        generate.add_packages(load_config(), int(options.repeat))
    elif options.mode == "download_data":
        from pgportfolio.marketdata.datamatrices import DataMatrices
        with open("./pgportfolio/net_config.json") as file:
            config = json.load(file)
        config = preprocess_config(config)
        start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
        end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
        DataMatrices(start=start,
                     end=end,
                     feature_number=config["input"]["feature_number"],
                     window_size=config["input"]["window_size"],
                     online=True,
                     period=config["input"]["global_period"],
                     volume_average_days=config["input"]["volume_average_days"],
                     coin_filter=config["input"]["coin_number"],
                     is_permed=config["input"]["is_permed"],
                     test_portion=config["input"]["test_portion"],
                     portion_reversed=config["input"]["portion_reversed"])
    elif options.mode == "backtest":
        config = _config_by_algo(options.algo)
        _set_logging_by_algo(logging.DEBUG, logging.DEBUG, options.algo, "backtestlog")
        execute_backtest(options.algo, config)
    elif options.mode == "save_test_data":
        # This is used to export the test data
        save_test_data(load_config(options.folder))
    elif options.mode == "plot":
        logging.basicConfig(level=logging.INFO)
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.plot_backtest(load_config(), algos, labels)
    elif options.mode == "table":
        algos = options.algos.split(",")
        if options.labels:
            labels = options.labels.replace("_"," ")
            labels = labels.split(",")
        else:
            labels = algos
        plot.table_backtest(load_config(), algos, labels, format=options.format)


def handle_data_commands():
    """处理数据管理命令"""
    if len(sys.argv) < 2:
        print("❌ 请指定数据管理命令")
        print("可用命令: data-create, data-status, data-download, data-download-all, data-complete, data-verify, data-monitor")
        sys.exit(1)

    command = sys.argv[1]

    # 创建数据管理器
    database_name = "DataNew.db"
    force = "--force" in sys.argv or "-f" in sys.argv

    # 解析数据库参数
    for i, arg in enumerate(sys.argv):
        if arg in ["--database", "-d"] and i + 1 < len(sys.argv):
            database_name = sys.argv[i + 1]
            break

    manager = BinanceDataManager(database_name)

    if command == "data-create":
        create_database(manager)
    elif command == "data-status":
        check_status(manager)
    elif command == "data-download":
        handle_download_single(manager, sys.argv[2:])
    elif command == "data-download-all":
        download_all(manager, force)
    elif command == "data-complete":
        complete_missing(manager)
    elif command == "data-verify":
        verify_database(manager)
    elif command == "data-monitor":
        monitor_progress(manager)
    else:
        print(f"❌ 未知命令: {command}")
        print("可用命令: data-create, data-status, data-download, data-download-all, data-complete, data-verify, data-monitor")
        sys.exit(1)


def create_database(manager):
    """创建数据库"""
    print("🔧 创建数据库...")
    manager._ensure_database_exists()

    is_valid, message = manager.verify_database()
    if is_valid:
        print("✅ 数据库创建成功")
        print(f"📁 数据库路径: {manager.db_path}")
    else:
        print(f"❌ 数据库创建失败: {message}")


def check_status(manager):
    """检查数据状态"""
    print("📊 检查数据库状态...")

    status = manager.get_download_status()

    if not status['exists']:
        print("❌ 数据库文件不存在")
        return

    print(f"📁 数据库路径: {manager.db_path}")
    print(f"📈 总记录数: {status['total_records']:,}")
    print(f"🪙 已下载币种: {len(status['coins'])}/{len(manager.coins)}")

    if status['missing_coins']:
        print(f"❌ 缺失币种: {', '.join(status['missing_coins'])}")

    if status['incomplete_coins']:
        incomplete_str = ', '.join([f"{coin}({count:,})" for coin, count in status['incomplete_coins']])
        print(f"⚠️  不完整币种: {incomplete_str}")

    if status['coins']:
        print(f"\n📋 各币种数据量:")
        expected_count = 365 * 3 * 288
        for coin, data in sorted(status['coins'].items(), key=lambda x: x[1]['count'], reverse=True):
            completeness = min((data['count'] / expected_count) * 100, 100)
            status_icon = "✅" if completeness >= 95 else "⚠️" if completeness >= 50 else "❌"
            print(f"  {status_icon} {coin:<8} {data['count']:>8,} 条 ({completeness:>5.1f}%)")


def handle_download_single(manager, args):
    """处理单个币种下载命令"""
    if not args:
        print("❌ 请指定要下载的币种")
        print("用法: python3 main.py data-download <币种名称> [--append]")
        sys.exit(1)

    coin = args[0]
    append = "--append" in args

    if coin not in manager.coins:
        print(f"❌ 不支持的币种: {coin}")
        print(f"支持的币种: {', '.join(manager.coins.keys())}")
        return

    print(f"📥 开始下载 {coin}...")
    success = manager.download_coin(coin, append_mode=append)

    if success:
        print(f"✅ {coin} 下载成功")
    else:
        print(f"❌ {coin} 下载失败")


def download_all(manager, force):
    """下载所有币种数据"""
    print("🚀 开始下载所有币种数据...")

    # 显示下载信息
    print(f"📊 下载信息:")
    print(f"  币种数量: {len(manager.coins)}")
    print(f"  时间范围: {manager.config.get('start_date', '2022-01-01')} 到 {manager.config.get('end_date', '2025-01-01')}")
    print(f"  时间间隔: {manager.config.get('timeframe', '5m')}")
    print(f"  数据源: Binance API")

    print(f"\n🪙 将下载以下币种:")
    for i, coin in enumerate(manager.coins.keys(), 1):
        info = manager.coins[coin]
        print(f"  {i:2d}. {coin} ({info['name']}) - {info['symbol']}")

    # 确认开始
    if not force:
        confirm = input(f"\n确认开始下载所有 {len(manager.coins)} 个币种? (输入 'YES' 继续): ").strip()
        if confirm != 'YES':
            print("❌ 取消下载")
            return

    print(f"\n🎯 开始下载...")
    start_time = datetime.now()

    # 执行下载
    success = manager.download_all_coins()

    # 显示结果
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'='*60}")
    print(f"{'🎉 下载完成!' if success else '⚠️  部分完成'}")
    print(f"⏱️  总耗时: {duration}")

    # 显示最终统计
    final_status = manager.get_download_status()
    print(f"📊 最终统计:")
    print(f"  总记录数: {final_status['total_records']:,}")
    print(f"  完成币种: {len(final_status['coins'])}/{len(manager.coins)}")


def complete_missing(manager):
    """补全缺失数据"""
    print("🔧 开始补全缺失数据...")

    success = manager.complete_missing_data()

    if success:
        print("✅ 补全完成")
        # 显示最终状态
        final_status = manager.get_download_status()
        print(f"📊 最终统计:")
        print(f"  总记录数: {final_status['total_records']:,}")
        print(f"  完成币种: {len(final_status['coins'])}/{len(manager.coins)}")
    else:
        print("❌ 补全过程中出现错误")


def verify_database(manager):
    """验证数据库"""
    print("🔍 验证数据库...")

    is_valid, message = manager.verify_database()

    if is_valid:
        print("✅ 数据库验证通过")

        # 显示数据库信息
        status = manager.get_download_status()
        if status['exists']:
            print(f"📊 数据库信息:")
            print(f"  路径: {manager.db_path}")
            print(f"  记录数: {status['total_records']:,}")
            print(f"  币种数: {len(status['coins'])}")
    else:
        print(f"❌ 数据库验证失败: {message}")


def monitor_progress(manager):
    """监控下载进度"""
    print("📊 启动下载进度监控...")

    try:
        while True:
            # 清屏
            os.system('cls' if os.name == 'nt' else 'clear')

            print("📊 数据下载进度监控")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            status = manager.get_download_status()

            if not status['exists']:
                print("🔄 数据库文件未找到，可能正在创建...")
                time.sleep(5)
                continue

            expected_per_coin = 365 * 3 * 288
            expected_total = expected_per_coin * len(manager.coins)
            progress_percent = min((status['total_records'] / expected_total) * 100, 100)

            print(f"📈 总进度: {progress_percent:.1f}% ({status['total_records']:,} / {expected_total:,})")

            if status['total_records'] > 0:
                print(f"\n🪙 各币种下载进度:")
                for coin, data in sorted(status['coins'].items(), key=lambda x: x[1]['count'], reverse=True):
                    coin_progress = min((data['count'] / expected_per_coin) * 100, 100)

                    # 进度条
                    bar_length = 20
                    filled_length = int(bar_length * coin_progress / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)

                    print(f"  {coin:<8} [{bar}] {coin_progress:>5.1f}%")
                    print(f"          {data['count']:>8,}/{expected_per_coin:,} 条")

            print(f"\n" + "-" * 40)
            print("按 Ctrl+C 停止监控 (10秒后刷新)")

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\n⏹️  停止监控")
        final_status = manager.get_download_status()
        print(f"✅ 监控结束，当前总记录数: {final_status['total_records']:,}")

def _set_logging_by_algo(console_level, file_level, algo, name):
    if algo.isdigit():
            logging.basicConfig(filename="./train_package/"+algo+"/"+name,
                                level=file_level)
            console = logging.StreamHandler()
            console.setLevel(console_level)
            logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(level=console_level)


def _config_by_algo(algo):
    """
    :param algo: a string represent index or algo name
    :return : a config dictionary
    """
    if not algo:
        raise ValueError("please input a specific algo")
    elif algo.isdigit():
        config = load_config(algo)
    else:
        config = load_config()
    return config

if __name__ == "__main__":
    main()
