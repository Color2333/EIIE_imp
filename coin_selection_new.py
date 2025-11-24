"""
币种配置文件 - 基于2024-2025年主流加密货币市场
选择的币种具有良好的流动性、市场地位和代表性
"""

# 推荐的15个主流币种（按市值和影响力排序）
RECOMMENDED_COINS = {
    # Layer 1 公链
    'ETH': {'symbol': 'ETHBTC', 'name': 'Ethereum', 'category': 'Layer 1', 'rank': 2},
    'BNB': {'symbol': 'BNBBTC', 'name': 'Binance Coin', 'category': 'Layer 1', 'rank': 4},
    'SOL': {'symbol': 'SOLBTC', 'name': 'Solana', 'category': 'Layer 1', 'rank': 5},
    'AVAX': {'symbol': 'AVAXBTC', 'name': 'Avalanche', 'category': 'Layer 1', 'rank': 10},
    'DOT': {'symbol': 'DOTBTC', 'name': 'Polkadot', 'category': 'Layer 1', 'rank': 14},

    # DeFi 生态
    'LINK': {'symbol': 'LINKBTC', 'name': 'Chainlink', 'category': 'DeFi', 'rank': 15},
    'UNI': {'symbol': 'UNIBTC', 'name': 'Uniswap', 'category': 'DeFi', 'rank': 18},

    # 支付和存储
    'XRP': {'symbol': 'XRPBTC', 'name': 'Ripple', 'category': 'Payment', 'rank': 7},
    'ADA': {'symbol': 'ADABTC', 'name': 'Cardano', 'category': 'Layer 1', 'rank': 8},
    'DOGE': {'symbol': 'DOGEBTC', 'name': 'Dogecoin', 'category': 'Payment', 'rank': 9},
    'LTC': {'symbol': 'LTCBTC', 'name': 'Litecoin', 'category': 'Payment', 'rank': 20},

    # 其他重要币种
    'MATIC': {'symbol': 'MATICBTC', 'name': 'Polygon', 'category': 'Layer 2', 'rank': 13},
    'ATOM': {'symbol': 'ATOMBTC', 'name': 'Cosmos', 'category': 'Layer 1', 'rank': 47},
    'FTM': {'symbol': 'FTMBTC', 'name': 'Fantom', 'category': 'Layer 1', 'rank': 53},
    'SAND': {'symbol': 'SANDBTC', 'name': 'The Sandbox', 'category': 'Metaverse', 'rank': 54},

    # 稳定币 (需要特殊处理)
    'reversed_USDT': {'symbol': 'BTCUSDT', 'name': 'Tether (reversed)', 'category': 'Stablecoin', 'rank': 3, 'is_reversed': True}
}

def get_database_config(auto_time=False, use_config=False):
    """
    获取数据库配置，支持智能时间管理

    Args:
        auto_time: 是否自动使用当前时间
        use_config: 是否从配置文件读取时间范围

    Returns:
        dict: 数据库配置
    """
    from datetime import datetime, timedelta

    # 基础配置
    config = {
        'name': 'DataNew.db',
        'timeframe': '5m',  # 5分钟间隔
        'start_date': '2022-01-01',  # 默认开始日期
        'end_date': '2025-01-01',    # 默认结束日期
        'base_currency': 'BTC',      # 以BTC为计价单位
        'data_source': 'Binance API', # 数据源
        'features': ['close', 'high', 'low', 'open', 'volume'],  # 数据特征
    }

    # 自动时间模式：使用当前日期
    if auto_time:
        current_date = datetime.now()
        config['end_date'] = current_date.strftime('%Y-%m-%d')
        print(f"🕒 自动时间模式：数据范围设置为 {config['start_date']} 到 {config['end_date']}")

    # 从配置文件读取时间范围
    elif use_config:
        try:
            # 查找最新的训练包配置
            import json
            import os

            config_paths = [
                './pgportfolio/net_config.json',
                './train_package/*/net_config.json'
            ]

            end_date = None
            start_date = None

            # 首先尝试主配置文件
            main_config = './pgportfolio/net_config.json'
            if os.path.exists(main_config):
                with open(main_config, 'r') as f:
                    net_config = json.load(f)
                    end_date = net_config.get('input', {}).get('end_date')
                    start_date = net_config.get('input', {}).get('start_date')

            # 如果主配置文件没有，尝试训练包
            if end_date is None or start_date is None:
                train_packages = glob.glob('./train_package/*/net_config.json')
                if train_packages:
                    # 按修改时间排序，使用最新的配置
                    latest_config = max(train_packages, key=os.path.getmtime)
                    with open(latest_config, 'r') as f:
                        net_config = json.load(f)
                        end_date = net_config.get('input', {}).get('end_date')
                        start_date = net_config.get('input', {}).get('start_date')

            # 应用找到的时间配置
            if end_date:
                # 转换格式: 2024/12/01 -> 2024-12-01
                config['end_date'] = end_date.replace('/', '-')
                print(f"📋 从配置文件读取结束日期：{config['end_date']}")

            if start_date:
                config['start_date'] = start_date.replace('/', '-')
                print(f"📋 从配置文件读取开始日期：{config['start_date']}")

            if end_date or start_date:
                print(f"📊 配置时间范围：{config['start_date']} 到 {config['end_date']}")
            else:
                print("⚠️  未在配置文件中找到时间范围，使用默认值")

        except Exception as e:
            print(f"⚠️  读取配置文件失败：{e}")
            print("🔄 使用默认时间范围")

    return config

# 默认数据库配置
DATABASE_CONFIG = get_database_config()

# 时间间隔映射（Binance API）
BINANCE_INTERVALS = {
    '1m': '1m',
    '3m': '3m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '2h': '2h',
    '4h': '4h',
    '6h': '6h',
    '8h': '8h',
    '12h': '12h',
    '1d': '1d',
    '3d': '3d',
    '1w': '1w',
    '1M': '1M'
}

# Binance API配置
BINANCE_CONFIG = {
    'base_url': 'https://api.binance.com/api/v3/klines',
    'rate_limit': 1200,  # 每分钟请求限制
    'retry_attempts': 3,
    'timeout': 30
}

# 数据质量检查配置
DATA_QUALITY = {
    'min_price': 1e-8,      # 最小价格阈值
    'max_price': 1000,      # 最大价格阈值（相对于BTC）
    'min_volume': 0.01,     # 最小交易量
    'max_gap_ratio': 0.5,   # 最大价格跳跃比例
    'required_data_points': 2000  # 每个币种至少需要的数据点
}

# 获取所有选择的币种列表
def get_selected_coins():
    """获取所有选择的币种"""
    return list(RECOMMENDED_COINS.keys())

# 获取币种对映射
def get_coin_pairs():
    """获取币种对映射字典"""
    return {coin: info['symbol'] for coin, info in RECOMMENDED_COINS.items()}

# 获取币种信息
def get_coin_info(coin):
    """获取特定币种信息"""
    return RECOMMENDED_COINS.get(coin, None)

# 按类别获取币种
def get_coins_by_category(category):
    """按类别获取币种"""
    return [coin for coin, info in RECOMMENDED_COINS.items()
            if info['category'] == category]

# 获取前N个币种
def get_top_coins(n=10):
    """获取排名前N的币种"""
    sorted_coins = sorted(RECOMMENDED_COINS.items(),
                         key=lambda x: x[1]['rank'])
    return [coin[0] for coin in sorted_coins[:n]]

if __name__ == "__main__":
    # 测试配置
    print("选择的币种数量:", len(RECOMMENDED_COINS))
    print("币种列表:", get_selected_coins())
    print("币种对映射:", get_coin_pairs())
    print("\n按类别分类:")
    for category in set(info['category'] for info in RECOMMENDED_COINS.values()):
        coins = get_coins_by_category(category)
        print(f"{category}: {coins}")