# 数据库切换使用指南

本指南说明如何在 `data` 和 `data2` 数据库之间进行切换。

## 数据库信息

### data 数据库（默认）
- **时间范围**: 2015-07-01 到 2017-07-01
- **币种数量**: 11 个
- **包含币种**: DASH, ETC, ETH, FCT, GNT, LTC, XEM, XMR, XRP, ZEC, reversed_USDT
- **数据库文件**: `database/Data.db`

### data2 数据库
- **时间范围**: 2022-01-01 到 2024-12-31
- **币种数量**: 11 个
- **包含币种**: ADA, AVAX, BNB, DOGE, DOT, ETH, LINK, LTC, SOL, XRP, reversed_USDT
- **数据库文件**: `database/Data2.db`

## 使用方法

### 方法1: 使用交互式切换工具（推荐）

```bash
python3 switch_database.py
```

这个工具会：
1. 显示当前数据库信息
2. 列出所有可用数据库
3. 让您选择要切换的数据库
4. 验证数据库文件是否存在
5. 提供配置建议

### 方法2: 直接使用Python代码

```python
import sys
sys.path.append('.')
from pgportfolio.database_config import switch_database, get_database_info

# 切换到 data2 数据库
if switch_database('data2'):
    print('切换成功!')
    info = get_database_info()
    print(f'当前数据库: {info["name"]}')
    print(f'包含币种: {info["coins"]}')
else:
    print('切换失败!')

# 切换回 data 数据库
if switch_database('data'):
    print('已切换回 data 数据库')
```

### 方法3: 手动编辑配置文件

直接编辑 `database_config.json` 文件：

```json
{
  "current_database": "data2",
  "databases": {
    "data": {
      "name": "Original Database (2015-2017)",
      "file": "database/Data.db",
      ...
    },
    "data2": {
      "name": "Modern Database (2022-2024)",
      "file": "database/Data2.db",
      ...
    }
  }
}
```

## 配置文件说明

配置文件 `database_config.json` 位于项目根目录，包含：
- `current_database`: 当前使用的数据库标识符
- `databases`: 数据库配置字典

## 注意事项

### 1. 网络配置
切换数据库后，请检查 `pgportfolio/net_config.json` 中的 `coin_number` 配置：

```json
{
  "input_features": [...],
  "coin_number": 11,
  ...
}
```

两个数据库都有11个币种，所以 `coin_number` 可以保持为 11。

### 2. 重新训练
切换数据库后，如果需要重新训练模型：
1. 删除之前的训练结果：`rm -rf train_package/*`
2. 重新运行训练命令

### 3. 验证数据库文件
确保数据库文件存在于 `database/` 目录：
- `Data.db` (原始数据库)
- `Data2.db` (现代数据库)

### 4. 币种差异
虽然两个数据库都有11个币种，但具体币种不同：
- data: 主要是较老的加密货币
- data2: 包含更多现代主流加密货币

## 故障排除

### 问题: 数据库文件不存在
**解决方案**: 确保数据库文件在正确位置
```bash
ls -la database/
# 应该看到 Data.db 和 Data2.db
```

### 问题: 切换后无法加载数据
**解决方案**:
1. 检查数据库文件是否损坏
2. 验证 `net_config.json` 中的配置
3. 重新启动训练程序

### 问题: 配置文件不存在
**解决方案**: 重新生成配置文件
```python
python3 -c "from pgportfolio.database_config import db_config; db_config.save_config()"
```

## 技术实现

数据库切换功能通过以下文件实现：
- `pgportfolio/database_config.py`: 数据库配置管理
- `pgportfolio/constants.py`: 常量定义，使用配置系统
- `switch_database.py`: 用户友好的切换工具
- `database_config.json`: 配置文件

这个系统允许灵活地管理和切换不同的数据库，而无需修改源代码。