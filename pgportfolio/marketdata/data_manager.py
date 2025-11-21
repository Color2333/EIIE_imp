"""
数据下载管理器
整合Binance数据下载、数据库管理和补全功能
"""

import sqlite3
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from .binance_api import BinanceAPI

# 配置日志
logger = logging.getLogger(__name__)


class BinanceDataManager:
    """
    Binance数据管理器
    统一管理数据下载、存储、补全等功能
    """

    def __init__(self, database_name: str = "DataNew.db"):
        """
        初始化数据管理器

        Args:
            database_name: 数据库名称
        """
        self.api = BinanceAPI()
        self.db_path = os.path.join("database", database_name)

        # 从配置获取币种信息
        try:
            from coin_selection_new import RECOMMENDED_COINS, DATABASE_CONFIG
            self.coins = RECOMMENDED_COINS
            self.config = DATABASE_CONFIG
        except ImportError:
            # 如果无法导入，使用默认配置
            self.coins = {}
            self.config = {
                'start_date': '2022-01-01',
                'end_date': '2025-01-01',
                'timeframe': '5m'
            }
            logger.warning("无法导入币种配置，使用默认设置")

        # 确保数据库目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def check_database_status(self) -> Dict:
        """
        检查数据库状态

        Returns:
            Dict: 数据库状态信息
        """
        try:
            if not os.path.exists(self.db_path):
                return {
                    'exists': False,
                    'total_records': 0,
                    'coins': {},
                    'missing_coins': list(self.coins.keys()),
                    'incomplete_coins': []
                }

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取总体统计
            cursor.execute("SELECT COUNT(*) FROM History;")
            total_records = cursor.fetchone()[0]

            # 获取每个币种的统计
            cursor.execute("""
                SELECT coin, COUNT(*) as count, MIN(date), MAX(date)
                FROM History
                GROUP BY coin
            """)
            db_results = cursor.fetchall()

            coins_data = {}
            for row in db_results:
                coin, count, min_date, max_date = row
                coins_data[coin] = {
                    'count': count,
                    'start_date': min_date,
                    'end_date': max_date
                }

            conn.close()

            expected_count = 365 * 3 * 288  # 316,800 条记录
            missing_coins = []
            incomplete_coins = []

            for coin in self.coins.keys():
                if coin not in coins_data:
                    missing_coins.append(coin)
                else:
                    count = coins_data[coin]['count']
                    if count < expected_count * 0.95:  # 95% 完整度
                        incomplete_coins.append((coin, count))

            return {
                'exists': True,
                'total_records': total_records,
                'coins': coins_data,
                'missing_coins': missing_coins,
                'incomplete_coins': incomplete_coins
            }

        except Exception as e:
            logger.error(f"检查数据库状态失败: {e}")
            return {
                'exists': False,
                'total_records': 0,
                'coins': {},
                'missing_coins': list(self.coins.keys()),
                'incomplete_coins': []
            }

    def download_coin_batched(self, coin_name: str, append_mode: bool = False, batch_months: int = 3) -> bool:
        """
        分批下载单个币种数据（专为大数据量设计，特别是reversed_USDT）

        Args:
            coin_name: 币种名称
            append_mode: 是否为追加模式
            batch_months: 每批处理的月份数（默认3个月）

        Returns:
            bool: 下载是否成功
        """
        if coin_name not in self.coins:
            logger.error(f"不支持的币种: {coin_name}")
            return False

        try:
            from datetime import datetime, timedelta
            import calendar

            start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')

            # 追加模式处理
            if append_mode and os.path.exists(self.db_path):
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT MAX(date) FROM History WHERE coin = ?",
                        (coin_name,)
                    )
                    result = cursor.fetchone()
                    conn.close()

                    if result and result[0]:
                        last_timestamp = result[0]
                        start_date = datetime.fromtimestamp(last_timestamp + 300)
                        logger.info(f"{coin_name} 追加模式：从 {start_date.strftime('%Y-%m-%d %H:%M:%S')} 开始下载")

                except Exception as e:
                    logger.warning(f"检查 {coin_name} 已有数据失败: {e}")

            logger.info(f"开始分批下载 {coin_name} ({'追加模式' if append_mode else '完整模式'})")

            # 确保数据库和表存在
            self._ensure_database_exists()

            total_batches = 0
            success_count = 0

            current_date = start_date
            batch_num = 1

            while current_date < end_date:
                # 计算批次结束日期
                batch_end_date = current_date + timedelta(days=batch_months * 30)
                if batch_end_date > end_date:
                    batch_end_date = end_date

                total_batches += 1

                # 转换为API需要的格式
                start_str = current_date.strftime('%Y-%m-%d')
                end_str = batch_end_date.strftime('%Y-%m-%d')

                logger.info(f"批次 {batch_num}: 下载 {start_str} 到 {end_str} 的数据")

                # 获取币种对符号
                symbol = self.coins[coin_name]['symbol']

                # 下载数据
                raw_data = self.api.get_historical_data(
                    symbol=symbol,
                    interval=self.config['timeframe'],
                    start_date=start_str,
                    end_date=end_str
                )

                if not raw_data:
                    logger.error(f"批次 {batch_num}: 没有获取到 {coin_name} 的数据")
                    current_date = batch_end_date
                    batch_num += 1
                    continue

                # 转换数据格式
                standard_data = self.api.convert_to_standard_format(raw_data, coin_name)

                if not standard_data:
                    logger.error(f"批次 {batch_num}: {coin_name} 数据转换失败")
                    current_date = batch_end_date
                    batch_num += 1
                    continue

                # 验证数据质量
                is_valid, message = self.api.validate_data_quality(standard_data)
                if not is_valid:
                    logger.error(f"批次 {batch_num}: {coin_name} 数据质量验证失败: {message}")
                    current_date = batch_end_date
                    batch_num += 1
                    continue

                # 存储这批数据
                if self._store_coin_data_batch(coin_name, standard_data):
                    logger.info(f"批次 {batch_num}: ✅ 成功存储 {len(standard_data)} 条记录")
                    success_count += 1
                else:
                    logger.error(f"批次 {batch_num}: ❌ 存储失败")

                # 移动到下一批次
                current_date = batch_end_date
                batch_num += 1

                # 添加短暂延迟，避免API频率限制
                import time
                time.sleep(1)

            logger.info(f"分批下载完成！成功: {success_count}/{total_batches} 批次")
            return success_count == total_batches

        except Exception as e:
            logger.error(f"分批下载 {coin_name} 时出错: {e}")
            return False

    def _store_coin_data_batch(self, coin_name: str, data: List[List]) -> bool:
        """
        分批存储币种数据到数据库

        Args:
            coin_name: 币种名称
            data: 数据列表

        Returns:
            bool: 存储是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 使用事务插入数据，每1000条提交一次
            cursor.execute("BEGIN TRANSACTION;")

            for i, row in enumerate(data):
                cursor.execute("""
                    INSERT OR REPLACE INTO History
                    (date, coin, high, low, open, close, volume, quoteVolume, weightedAverage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)

                # 每1000条提交一次，减少内存占用
                if i % 1000 == 0:
                    cursor.execute("COMMIT;")
                    cursor.execute("BEGIN TRANSACTION;")

            cursor.execute("COMMIT;")
            conn.close()

            logger.debug(f"成功存储 {coin_name} 批次数据 {len(data)} 条记录")
            return True

        except Exception as e:
            logger.error(f"存储 {coin_name} 批次数据失败: {e}")
            return False

    def download_coin(self, coin_name: str, append_mode: bool = False) -> bool:
        """
        下载单个币种数据

        Args:
            coin_name: 币种名称
            append_mode: 是否为追加模式

        Returns:
            bool: 下载是否成功
        """
        # 对于reversed_USDT，使用分批下载
        if coin_name == 'reversed_USDT':
            logger.info("检测到reversed_USDT，使用分批下载模式")
            return self.download_coin_batched(coin_name, append_mode, batch_months=2)  # 2个月一批，减少内存压力

        if coin_name not in self.coins:
            logger.error(f"不支持的币种: {coin_name}")
            return False

        try:
            start_date = self.config['start_date']
            end_date = self.config['end_date']

            # 追加模式处理
            if append_mode and os.path.exists(self.db_path):
                try:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    cursor.execute(
                        "SELECT MAX(date) FROM History WHERE coin = ?",
                        (coin_name,)
                    )
                    result = cursor.fetchone()
                    conn.close()

                    if result and result[0]:
                        last_timestamp = result[0]
                        start_date = datetime.fromtimestamp(last_timestamp + 300).strftime('%Y-%m-%d %H:%M:%S')
                        logger.info(f"{coin_name} 追加模式：从 {start_date} 开始下载")

                except Exception as e:
                    logger.warning(f"检查 {coin_name} 已有数据失败: {e}")

            logger.info(f"开始下载 {coin_name} ({'追加模式' if append_mode else '完整模式'})")

            # 获取币种对符号
            symbol = self.coins[coin_name]['symbol']

            # 下载数据
            raw_data = self.api.get_historical_data(
                symbol=symbol,
                interval=self.config['timeframe'],
                start_date=start_date,
                end_date=end_date
            )

            if not raw_data:
                logger.error(f"没有获取到 {coin_name} 的数据")
                return False

            # 转换数据格式
            standard_data = self.api.convert_to_standard_format(raw_data, coin_name)

            if not standard_data:
                logger.error(f"{coin_name} 数据转换失败")
                return False

            # 验证数据质量
            is_valid, message = self.api.validate_data_quality(standard_data)
            if not is_valid:
                logger.error(f"{coin_name} 数据质量验证失败: {message}")
                return False

            # 存储到数据库
            return self._store_coin_data(coin_name, standard_data)

        except Exception as e:
            logger.error(f"下载 {coin_name} 时出错: {e}")
            return False

    def _store_coin_data(self, coin_name: str, data: List[List]) -> bool:
        """
        存储币种数据到数据库

        Args:
            coin_name: 币种名称
            data: 数据列表

        Returns:
            bool: 存储是否成功
        """
        try:
            # 确保数据库和表存在
            self._ensure_database_exists()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 使用事务插入数据
            cursor.execute("BEGIN TRANSACTION;")

            for row in data:
                cursor.execute("""
                    INSERT OR REPLACE INTO History
                    (date, coin, high, low, open, close, volume, quoteVolume, weightedAverage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)

            cursor.execute("COMMIT;")
            conn.close()

            logger.info(f"成功存储 {coin_name} 数据 {len(data)} 条记录")
            return True

        except Exception as e:
            logger.error(f"存储 {coin_name} 数据失败: {e}")
            return False

    def _ensure_database_exists(self):
        """确保数据库和表存在"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 创建History表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS History (
                    date INTEGER NOT NULL,
                    coin VARCHAR(20) NOT NULL,
                    high FLOAT NOT NULL,
                    low FLOAT NOT NULL,
                    open FLOAT NOT NULL,
                    close FLOAT NOT NULL,
                    volume FLOAT NOT NULL,
                    quoteVolume FLOAT NOT NULL,
                    weightedAverage FLOAT NOT NULL,
                    PRIMARY KEY (date, coin)
                );
            """)

            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_coin ON History(coin);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON History(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_coin_date ON History(coin, date);")

            # 创建metadata表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                );
            """)

            conn.close()
            logger.debug("数据库结构检查完成")

        except Exception as e:
            logger.error(f"创建数据库结构失败: {e}")

    def download_all_coins(self, start_date: str = None, end_date: str = None) -> bool:
        """
        下载所有币种数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            bool: 是否全部成功
        """
        success_count = 0
        total_start_time = time.time()

        for i, coin_name in enumerate(self.coins.keys(), 1):
            logger.info(f"[{i}/{len(self.coins)}] 正在下载 {coin_name}")

            if self.download_coin(coin_name):
                success_count += 1
                logger.info(f"✅ {coin_name} 下载成功")
            else:
                logger.error(f"❌ {coin_name} 下载失败")

            # 避免API频率限制
            time.sleep(1)

        total_time = time.time() - total_start_time
        logger.info(f"\n下载完成！成功: {success_count}/{len(self.coins)}")
        logger.info(f"总耗时: {total_time:.2f} 秒")

        return success_count == len(self.coins)

    def complete_missing_data(self) -> bool:
        """
        补全缺失的数据

        Returns:
            bool: 补全是否成功
        """
        status = self.check_database_status()

        missing_coins = status['missing_coins']
        incomplete_coins = status['incomplete_coins']

        if not missing_coins and not incomplete_coins:
            logger.info("🎉 所有数据都已完整，无需补全")
            return True

        logger.info(f"开始补全数据:")
        logger.info(f"缺失币种: {missing_coins}")
        logger.info(f"不完整币种: {[(coin, count) for coin, count in incomplete_coins]}")

        # 下载缺失的币种
        for coin in missing_coins:
            try:
                logger.info(f"下载缺失币种: {coin}")
                success = self.download_coin(coin)
                if success:
                    logger.info(f"✅ {coin} 下载完成")
                else:
                    logger.error(f"❌ {coin} 下载失败")
            except Exception as e:
                logger.error(f"❌ {coin} 下载出错: {e}")

        # 补全不完整的币种
        for coin, current_count in incomplete_coins:
            try:
                logger.info(f"补全不完整币种: {coin}")
                success = self.download_coin(coin, append_mode=True)
                if success:
                    logger.info(f"✅ {coin} 补全完成")
                else:
                    logger.error(f"❌ {coin} 补全失败")
            except Exception as e:
                logger.error(f"❌ {coin} 补全出错: {e}")

        return True

    def get_download_status(self) -> Dict:
        """
        获取下载状态

        Returns:
            Dict: 下载状态信息
        """
        return self.check_database_status()

    def verify_database(self) -> Tuple[bool, str]:
        """
        验证数据库完整性

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            if not os.path.exists(self.db_path):
                return False, "数据库文件不存在"

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查表结构
            cursor.execute("PRAGMA table_info(History);")
            columns = [row[1] for row in cursor.fetchall()]

            required_columns = ['date', 'coin', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']
            for column in required_columns:
                if column not in columns:
                    conn.close()
                    return False, f"History表缺少必需的列: {column}"

            # 检查索引
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            index_rows = cursor.fetchall()
            indexes = [row[0] for row in index_rows if row[0] and row[0].startswith('idx_')]

            required_indexes = ['idx_coin', 'idx_date', 'idx_coin_date']
            for index in required_indexes:
                if index not in indexes:
                    logger.warning(f"缺少推荐的索引: {index}")

            # 检查数据质量
            cursor.execute("SELECT COUNT(*) FROM History WHERE high <= 0 OR low <= 0;")
            bad_records = cursor.fetchone()[0]

            conn.close()

            if bad_records > 0:
                return False, f"发现 {bad_records} 条异常记录"

            return True, "数据库验证通过"

        except Exception as e:
            return False, f"验证数据库失败: {e}"