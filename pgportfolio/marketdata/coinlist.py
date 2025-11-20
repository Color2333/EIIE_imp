from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from pgportfolio.marketdata.poloniex import Poloniex
from pgportfolio.tools.data import get_chart_until_success
import pandas as pd
from datetime import datetime
import logging
from pgportfolio.constants import *
import numpy as np


class CoinList(object):
    def __init__(self, end, volume_average_days=1, volume_forward=0):
        self._polo = Poloniex()

        # 尝试从数据库选择币种，如果失败则尝试API
        try:
            self._select_from_database(end, volume_average_days, volume_forward)
        except Exception as e:
            logging.warning(f"Database selection failed: {e}")
            logging.info("Falling back to API selection")
            self._select_from_api(end, volume_average_days, volume_forward)

    def _select_from_database(self, end, volume_average_days, volume_forward):
        """从数据库选择币种"""
        import sqlite3
        from pgportfolio.constants import DATABASE_DIR

        # 确保end是整数时间戳
        if isinstance(end, str):
            from pgportfolio.tools.configprocess import parse_time
            end = parse_time(end)

        logging.info("select coins offline from database from %s to %s" % (
            datetime.fromtimestamp(end-(DAY*volume_average_days)-volume_forward).strftime('%Y-%m-%d %H:%M'),
            datetime.fromtimestamp(end-volume_forward).strftime('%Y-%m-%d %H:%M')
        ))

        conn = sqlite3.connect(DATABASE_DIR)

        # 计算时间范围
        start_time = end - (volume_average_days * DAY) - volume_forward
        end_time = end - volume_forward

        # 获取交易量最高的币种
        query = """
        SELECT coin, SUM(volume) as total_volume, AVG(close) as avg_price
        FROM History
        WHERE date >= ? AND date < ? AND coin != 'reversed_USDT'
        GROUP BY coin
        ORDER BY total_volume DESC
        """

        result = conn.execute(query, (start_time, end_time)).fetchall()

        if not result:
            # 如果没有数据，使用默认币种
            logging.warning("No volume data found in database, using default coins")
            self.pairs = ["BTC_ETH", "BTC_LTC", "BTC_XRP"]
            self.coins = ["ETH", "LTC", "XRP"]
            self.volumes = [1000, 1000, 1000]
            self.prices = [0.02, 0.01, 0.00002]
        else:
            # 选择前N个币种
            top_n = min(len(result), 15)  # 确保有足够的币种
            self.coins = []
            self.volumes = []
            self.prices = []
            self.pairs = []

            for i, (coin, volume, price) in enumerate(result[:top_n]):
                self.coins.append(coin)
                self.volumes.append(volume)
                self.prices.append(price)
                self.pairs.append(f"BTC_{coin}")

        conn.close()

    def _select_from_api(self, end, volume_average_days, volume_forward):
        """原始的API选择方法（作为后备）"""
        # connect the internet to accees volumes
        vol = self._polo.marketVolume()
        ticker = self._polo.marketTicker()
        pairs = []
        coins = []
        volumes = []
        prices = []

        logging.info("select coin online from %s to %s" % (datetime.fromtimestamp(end-(DAY*volume_average_days)-
                                                                                  volume_forward).
                                                           strftime('%Y-%m-%d %H:%M'),
                                                           datetime.fromtimestamp(end-volume_forward).
                                                           strftime('%Y-%m-%d %H:%M')))
        for k, v in vol.items():
            if k.startswith("BTC_") or k.endswith("_BTC"):
                pairs.append(k)
        for pair in pairs:
            if pair.startswith("BTC_"):
                coin = pair[4:]
            else:
                coin = pair[:-4]
            volume = float(vol[pair]["BTC_" + coin])
            if volume > 0:
                coins.append(coin)
                volumes.append(volume)
                price = float(ticker[pair]["last"])
                prices.append(price)

        self.pairs = pairs
        self.coins = coins
        self.volumes = volumes
        self.prices = prices

    def select_top_volume(self, coin_number=10):
        """
        :param coin_number: the number of selected coins
        :return: list of coins
        """
        # if offline, the coin_list could be None
        if not self.coins or len(self.coins) == 0:
            # 使用默认币种
            logging.warning("No coins available, using default selection")
            return ["ETH", "LTC", "XRP", "ADA", "DOT", "LINK", "BNB", "SOL", "AVAX", "DOGE"][:coin_number]

        if coin_number > len(self.coins):
            logging.warning("too many coin required, returning all the coins")
            return self.coins

        # 如果是BTC市场，返回coin
        if self.pairs[0].startswith("BTC_"):
            result = [self.coins[i] for i in np.argsort(self.volumes)[::-1][:coin_number]]
            return result
        # 如果是USDT市场，返回coin
        elif self.pairs[0].endswith("_USDT"):
            result = [self.coins[i] for i in np.argsort(self.volumes)[::-1][:coin_number]]
            return result
        # 如果是其他市场
        else:
            return self.coins[:coin_number]

    def get_coins_volume(self):
        """
        :return: {coin_name: volume}
        """
        return dict(zip(self.coins, self.volumes))

    def get_coins_prices(self):
        """
        :return: {coin_name: price}
        """
        return dict(zip(self.coins, self.prices))
