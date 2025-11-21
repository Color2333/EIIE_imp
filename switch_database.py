#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库切换工具
用于轻松在 data 和 data2 数据库之间切换
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pgportfolio.database_config import (
    switch_database,
    list_available_databases,
    get_database_info,
    validate_current_database,
    db_config
)

def main():
    print("="*60)
    print("PGPortfolio 数据库切换工具")
    print("="*60)

    # 显示当前数据库信息
    current_info = get_database_info()
    print(f"\n当前数据库: {db_config.config.get('current_database', 'data')}")
    print(f"数据库名称: {current_info.get('name', 'Unknown')}")
    print(f"数据库文件: {current_info.get('file', 'Unknown')}")
    print(f"描述: {current_info.get('description', 'No description')}")
    print(f"时间范围: {current_info.get('time_range', 'Unknown')}")
    print(f"包含币种: {', '.join(current_info.get('coins', []))}")

    # 验证当前数据库文件是否存在
    if validate_current_database():
        print("✅ 数据库文件存在")
    else:
        print("❌ 数据库文件不存在")

    # 显示可用数据库
    print("\n" + "="*60)
    print("可用数据库:")
    available = list_available_databases()
    for i, (db_key, db_info) in enumerate(available.items(), 1):
        marker = "👉 [当前]" if db_key == db_config.config.get('current_database') else "   "
        print(f"{marker} {i}. {db_key}")
        print(f"      名称: {db_info.get('name', 'Unknown')}")
        print(f"      文件: {db_info.get('file', 'Unknown')}")
        print(f"      时间: {db_info.get('time_range', 'Unknown')}")
        print(f"      币种: {len(db_info.get('coins', []))} 个")
        print()

    # 切换数据库
    while True:
        try:
            choice = input("请选择要切换的数据库编号 (输入 q 退出): ").strip()
            if choice.lower() == 'q':
                print("退出数据库切换工具")
                break

            choice_num = int(choice)
            db_keys = list(available.keys())

            if 1 <= choice_num <= len(db_keys):
                selected_db = db_keys[choice_num - 1]
                print(f"\n正在切换到数据库: {selected_db}")

                if switch_database(selected_db):
                    print("✅ 数据库切换成功!")

                    # 验证切换后的数据库
                    if validate_current_database():
                        print("✅ 新数据库文件存在")
                    else:
                        print("⚠️  警告: 新数据库文件不存在，请检查文件路径")

                    # 显示新的配置信息
                    new_info = get_database_info()
                    print(f"\n当前数据库: {selected_db}")
                    print(f"数据库文件: {new_info.get('file', 'Unknown')}")
                    print(f"包含币种: {', '.join(new_info.get('coins', []))}")
                    print(f"时间范围: {new_info.get('time_range', 'Unknown')}")

                    # 提示需要更新网络配置
                    print(f"\n💡 提示:")
                    print(f"   - 切换数据库后，建议检查 pgportfolio/net_config.json 中的 coin_number 配置")
                    print(f"   - data 数据库有 11 个币种")
                    print(f"   - data2 数据库有 11 个币种")
                    print(f"   - 如果需要，可以手动修改 coin_number: {len(new_info.get('coins', []))}")

                    break
                else:
                    print("❌ 数据库切换失败!")
            else:
                print(f"❌ 无效选择，请输入 1-{len(db_keys)} 之间的数字")

        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n退出数据库切换工具")
            break

if __name__ == "__main__":
    main()