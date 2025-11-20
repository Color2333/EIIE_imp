#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库配置管理
支持多个数据库的切换和配置
"""

import os
import json
from os import path

class DatabaseConfig:
    """数据库配置管理类"""

    def __init__(self):
        self.config_file = "database_config.json"
        self.default_config = {
            "current_database": "data",
            "databases": {
                "data": {
                    "name": "Original Database (2015-2017)",
                    "file": "database/Data.db",
                    "description": "Original PGPortfolio database with 2015-2017 data",
                    "coins": ["DASH", "ETC", "ETH", "FCT", "GNT", "LTC", "XEM", "XMR", "XRP", "ZEC", "reversed_USDT"],
                    "time_range": "2015-07-01 to 2017-07-01"
                },
                "data2": {
                    "name": "Modern Database (2022-2024)",
                    "file": "database/Data2.db",
                    "description": "Updated database with 2022-2024 modern cryptocurrency data",
                    "coins": ["ADA", "AVAX", "BNB", "DOGE", "DOT", "ETH", "LINK", "LTC", "SOL", "XRP", "reversed_USDT"],
                    "time_range": "2022-01-01 to 2024-12-31"
                }
            }
        }
        self.load_config()

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                # 确保配置完整
                if "current_database" not in self.config:
                    self.config["current_database"] = "data"
                if "databases" not in self.config:
                    self.config["databases"] = self.default_config["databases"]
            except (json.JSONDecodeError, IOError):
                self.config = self.default_config.copy()
        else:
            self.config = self.default_config.copy()
            self.save_config()

    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError:
            pass

    def get_current_database_path(self):
        """获取当前数据库路径"""
        current_db = self.config.get("current_database", "data")
        db_info = self.config["databases"].get(current_db, self.config["databases"]["data"])

        # 获取相对于项目根目录的路径
        base_dir = path.dirname(path.dirname(path.abspath(__file__)))
        db_path = path.join(base_dir, db_info["file"])

        return db_path

    def set_current_database(self, db_name):
        """设置当前使用的数据库"""
        if db_name in self.config["databases"]:
            self.config["current_database"] = db_name
            self.save_config()
            return True
        return False

    def get_available_databases(self):
        """获取可用的数据库列表"""
        return self.config["databases"]

    def get_current_database_info(self):
        """获取当前数据库信息"""
        current_db = self.config.get("current_database", "data")
        return self.config["databases"].get(current_db, {})

    def validate_database(self, db_name=None):
        """验证数据库文件是否存在"""
        if db_name is None:
            db_name = self.config.get("current_database", "data")

        if db_name in self.config["databases"]:
            db_path = self.config["databases"][db_name]["file"]
            # 获取相对于项目根目录的路径
            base_dir = path.dirname(path.dirname(path.abspath(__file__)))
            full_path = path.join(base_dir, db_path)
            return os.path.exists(full_path)
        return False

# 创建全局数据库配置实例
db_config = DatabaseConfig()

def get_database_path():
    """获取当前数据库路径（用于兼容现有代码）"""
    return db_config.get_current_database_path()

def get_database_info():
    """获取当前数据库信息"""
    return db_config.get_current_database_info()

def list_available_databases():
    """列出所有可用数据库"""
    return db_config.get_available_databases()

def switch_database(db_name):
    """切换数据库"""
    return db_config.set_current_database(db_name)

def validate_current_database():
    """验证当前数据库"""
    return db_config.validate_database()