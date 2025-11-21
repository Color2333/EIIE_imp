#!/usr/bin/env python3
"""
数据下载命令行工具
整合了所有数据下载和管理功能
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pgportfolio.marketdata.data_manager import BinanceDataManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_database(args):
    """创建并初始化数据库"""
    print("🔧 创建数据库...")

    manager = BinanceDataManager(args.database)
    manager._ensure_database_exists()

    # 验证数据库
    is_valid, message = manager.verify_database()
    if is_valid:
        print("✅ 数据库创建成功")
        print(f"📁 数据库路径: {manager.db_path}")
    else:
        print(f"❌ 数据库创建失败: {message}")


def check_status(args):
    """检查下载状态"""
    print("📊 检查数据库状态...")

    manager = BinanceDataManager(args.database)
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


def download_all(args):
    """下载所有币种数据"""
    print("🚀 开始下载所有币种数据...")

    manager = BinanceDataManager(args.database)

    # 显示下载信息
    print(f"📊 下载信息:")
    print(f"  币种数量: {len(manager.coins)}")
    print(f"  时间范围: {manager.config['start_date']} 到 {manager.config['end_date']}")
    print(f"  时间间隔: {manager.config['timeframe']}")
    print(f"  数据源: Binance API")

    print(f"\n🪙 将下载以下币种:")
    for i, coin in enumerate(manager.coins.keys(), 1):
        info = manager.coins[coin]
        print(f"  {i:2d}. {coin} ({info['name']}) - {info['symbol']}")

    # 确认开始
    if not args.force:
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


def download_coin(args):
    """下载单个币种"""
    manager = BinanceDataManager(args.database)

    if args.coin not in manager.coins:
        print(f"❌ 不支持的币种: {args.coin}")
        print(f"支持的币种: {', '.join(manager.coins.keys())}")
        return

    print(f"📥 开始下载 {args.coin}...")
    success = manager.download_coin(args.coin, append_mode=args.append)

    if success:
        print(f"✅ {args.coin} 下载成功")
    else:
        print(f"❌ {args.coin} 下载失败")


def complete_missing(args):
    """补全缺失数据"""
    print("🔧 开始补全缺失数据...")

    manager = BinanceDataManager(args.database)
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


def verify(args):
    """验证数据库"""
    print("🔍 验证数据库...")

    manager = BinanceDataManager(args.database)
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


def monitor(args):
    """监控下载进度"""
    print("📊 启动下载进度监控...")

    manager = BinanceDataManager(args.database)

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

            import time
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\n⏹️  停止监控")
        final_status = manager.get_download_status()
        print(f"✅ 监控结束，当前总记录数: {final_status['total_records']:,}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PGPortfolio 数据下载管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s create                    # 创建数据库
  %(prog)s status                    # 检查状态
  %(prog)s download-all              # 下载所有币种
  %(prog)s download BTC              # 下载单个币种
  %(prog)s download DOT --append     # 追加下载DOT币种
  %(prog)s complete                  # 补全缺失数据
  %(prog)s monitor                   # 监控下载进度
  %(prog)s verify                    # 验证数据库
        """
    )

    parser.add_argument(
        '--database', '-d',
        default='DataNew.db',
        help='数据库文件名 (默认: DataNew.db)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='跳过确认提示'
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 创建数据库
    subparsers.add_parser('create', help='创建并初始化数据库').set_defaults(func=create_database)

    # 检查状态
    subparsers.add_parser('status', help='检查下载状态').set_defaults(func=check_status)

    # 下载所有币种
    subparsers.add_parser('download-all', help='下载所有币种数据').set_defaults(func=download_all)

    # 下载单个币种
    download_parser = subparsers.add_parser('download', help='下载单个币种数据')
    download_parser.add_argument('coin', help='币种名称')
    download_parser.add_argument('--append', action='store_true', help='追加模式')
    download_parser.set_defaults(func=download_coin)

    # 补全数据
    subparsers.add_parser('complete', help='补全缺失的数据').set_defaults(func=complete_missing)

    # 验证数据库
    subparsers.add_parser('verify', help='验证数据库完整性').set_defaults(func=verify)

    # 监控进度
    subparsers.add_parser('monitor', help='监控下载进度').set_defaults(func=monitor)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⏹️  操作已取消")
    except Exception as e:
        logger.error(f"执行命令时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()