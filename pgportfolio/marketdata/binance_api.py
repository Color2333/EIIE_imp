"""
Binance API 数据下载模块
用于从Binance获取历史K线数据，替代原有的Poloniex API
"""

import json
import time
import requests
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceAPI:
    """
    Binance API 客户端
    提供历史K线数据下载功能
    """

    def __init__(self, base_url: str = "https://api.binance.com/api/v3/klines"):
        """
        初始化Binance API客户端

        Args:
            base_url: Binance API基础URL
        """
        self.base_url = base_url
        self.rate_limit_delay = 0.1  # 请求间隔，避免触发频率限制
        self.retry_attempts = 3
        self.timeout = 30

        # 时间间隔常量
        self.minute = 60
        self.hour = 60 * self.minute
        self.day = 24 * self.hour
        self.week = 7 * self.day

    def get_klines(self,
                   symbol: str,
                   interval: str = "5m",
                   start_time: Optional[int] = None,
                   end_time: Optional[int] = None,
                   limit: int = 1000) -> List[List]:
        """
        获取K线数据

        Args:
            symbol: 交易对符号，如 'BTCUSDT', 'ETHBTC'
            interval: 时间间隔，如 '1m', '5m', '1h', '1d'
            start_time: 开始时间戳（毫秒）
            end_time: 结束时间戳（毫秒）
            limit: 限制返回条数（最大1000）

        Returns:
            List[List]: K线数据列表
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(self.base_url, params=params, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()

                # 检查是否返回错误信息
                if isinstance(data, dict) and 'code' in data:
                    raise Exception(f"Binance API Error: {data['msg']}")

                logger.info(f"成功获取 {symbol} {interval} 数据，条数: {len(data)}")
                return data

            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # 指数退避

            except Exception as e:
                logger.error(f"获取数据时出错: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.rate_limit_delay)

        raise Exception(f"获取 {symbol} 数据失败，已重试 {self.retry_attempts} 次")

    def get_historical_data(self,
                           symbol: str,
                           interval: str = "5m",
                           start_date: str = "2022-01-01",
                           end_date: str = "2025-01-01") -> List[List]:
        """
        获取历史数据（分批获取）

        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            List[List]: 完整的历史数据
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_data = []
        current_start = start_ts

        logger.info(f"开始下载 {symbol} 从 {start_date} 到 {end_date} 的 {interval} 数据")

        while current_start < end_ts:
            try:
                # 获取一批数据
                batch_data = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_ts,
                    limit=1000
                )

                if not batch_data:
                    logger.warning(f"没有获取到数据，停止下载 {symbol}")
                    break

                all_data.extend(batch_data)

                # 更新下次请求的开始时间
                last_timestamp = batch_data[-1][0]
                current_start = last_timestamp + 1

                # 进度显示
                progress = (current_start - start_ts) / (end_ts - start_ts) * 100
                logger.info(f"{symbol} 下载进度: {progress:.1f}% ({len(all_data)} 条数据)")

                # 请求间隔，避免频率限制
                time.sleep(self.rate_limit_delay)

                # 如果返回的数据少于限制数量，说明已经到达结束
                if len(batch_data) < 1000:
                    break

            except Exception as e:
                logger.error(f"下载过程中出错: {e}")
                break

        logger.info(f"完成下载 {symbol}，总共 {len(all_data)} 条数据")
        return all_data

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        获取最新价格

        Args:
            symbol: 交易对符号

        Returns:
            Optional[float]: 最新价格，获取失败返回None
        """
        try:
            data = self.get_klines(symbol=symbol, interval="1m", limit=1)
            if data:
                return float(data[0][4])  # 收盘价
        except Exception as e:
            logger.error(f"获取 {symbol} 最新价格失败: {e}")
        return None

    def convert_to_standard_format(self,
                                   kline_data: List[List],
                                   coin_name: str) -> List[List]:
        """
        将Binance K线数据转换为项目标准格式

        Args:
            kline_data: Binance原始K线数据
            coin_name: 币种名称

        Returns:
            List[List]: 标准格式数据 [date, coin, high, low, open, close, volume, quoteVolume, weightedAverage]
        """
        standard_data = []

        for kline in kline_data:
            # Binance数据格式: [open_time, open, high, low, close, volume, close_time,
            #                  quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
            #                  taker_buy_quote_asset_volume, ignore]

            date = int(kline[0] // 1000)  # 转换为秒时间戳
            open_price = float(kline[1])
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            volume = float(kline[5])
            quote_volume = float(kline[7])

            # 计算加权平均价格
            if quote_volume > 0:
                weighted_average = quote_volume / volume
            else:
                weighted_average = (high_price + low_price + close_price) / 3

            # 特殊处理reversed_USDT（与原项目保持一致的处理方式）
            if coin_name == 'reversed_USDT':
                # 对于reversed_USDT，需要做倒数处理
                # 因为BTCUSDT的数据是BTC相对于USDT的价格
                # 我们需要转换为USDT相对于BTC的价格（即reversed_USDT）

                # 注意：价格反转后，原来的最高价变成最低价，原来的最低价变成最高价
                original_high = high_price
                original_low = low_price

                if original_high > 0:
                    high_price = 1.0 / original_low  # 原最低价变成最高价
                if original_low > 0:
                    low_price = 1.0 / original_high  # 原最高价变成最低价
                if open_price > 0:
                    open_price = 1.0 / open_price
                if close_price > 0:
                    close_price = 1.0 / close_price
                if weighted_average > 0:
                    weighted_average = 1.0 / weighted_average
                # volume和quoteVolume需要交换
                volume, quote_volume = quote_volume, volume

            # 转换为项目存储格式
            standard_data.append([
                date, coin_name, high_price, low_price, open_price,
                close_price, volume, quote_volume, weighted_average
            ])

        return standard_data

    def validate_data_quality(self, data: List[List]) -> Tuple[bool, str]:
        """
        验证数据质量

        Args:
            data: 要验证的数据

        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        if not data:
            return False, "数据为空"

        # 检查数据点数量
        if len(data) < 100:
            return False, f"数据点数量过少: {len(data)}"

        # 检查价格合理性
        for i, row in enumerate(data):
            try:
                date, coin, high, low, open_price, close, volume, quote_volume, weighted_avg = row

                # 基本价格检查
                if high <= 0 or low <= 0 or open_price <= 0 or close <= 0:
                    return False, f"第{i}行存在非正价格"

                if high < low:
                    return False, f"第{i}行最高价低于最低价"

                if max(high, low, open_price, close) / min(high, low, open_price, close) > 100:
                    return False, f"第{i}行价格波动异常"

                # 检查时间戳连续性
                if i > 0:
                    time_diff = row[0] - data[i-1][0]
                    if time_diff < 0:
                        return False, f"第{i}行时间戳倒序"

            except Exception as e:
                return False, f"第{i}行数据格式错误: {e}"

        return True, "数据质量检查通过"

# 测试函数
if __name__ == "__main__":
    # 测试API连接
    api = BinanceAPI()

    try:
        # 测试获取少量数据
        test_data = api.get_klines(symbol="BTCUSDT", interval="1h", limit=5)
        print("API连接测试成功!")
        print(f"获取到 {len(test_data)} 条数据")

        # 测试数据转换
        standard_data = api.convert_to_standard_format(test_data, "BTC")
        print(f"转换后的标准格式数据示例:")
        if standard_data:
            print(standard_data[0])

        # 测试数据质量验证
        is_valid, message = api.validate_data_quality(standard_data)
        print(f"数据质量验证: {is_valid}, {message}")

    except Exception as e:
        print(f"API连接测试失败: {e}")