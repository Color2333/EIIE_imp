#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance Data Downloader - PGPortfolio Project
Download cryptocurrency historical data from Binance API and store to SQLite database
Supports complete download, incremental updates, resume functionality, etc.
"""

import os
import sys
import time
import sqlite3
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pgportfolio.marketdata.binance_api import BinanceAPI
from pgportfolio.marketdata.data_manager import BinanceDataManager
from coin_selection_new import RECOMMENDED_COINS, DATABASE_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BinanceDataDownloader:
    """
    Main Binance Data Downloader Class
    Provides complete download, management, and validation functionality
    """

    def __init__(self, database_name: str = "DataNew.db"):
        """
        Initialize downloader

        Args:
            database_name: Database name
        """
        self.api = BinanceAPI()
        self.manager = BinanceDataManager(database_name)
        self.coins = RECOMMENDED_COINS
        self.config = DATABASE_CONFIG

        logger.info(f"Initialize downloader, database: {database_name}")
        logger.info(f"Supported coins: {len(self.coins)}")

    def print_welcome(self):
        """Print welcome message"""
        print("=" * 60)
        print("🚀 PGPortfolio Binance Data Downloader")
        print("=" * 60)
        print(f"📊 Data Source: Binance API")
        print(f"📅 Time Range: {self.config['start_date']} to {self.config['end_date']}")
        print(f"⏱️  Time Interval: {self.config['timeframe']}")
        print(f"🪙 Coin Count: {len(self.coins)}")
        print(f"💾 Database: {self.manager.db_path}")
        print("=" * 60)

    def test_api_connection(self) -> bool:
        """Test API connection"""
        print("🔍 Testing Binance API connection...")

        try:
            # Get BTCUSDT latest price as test
            price = self.api.get_latest_price("BTCUSDT")
            if price:
                print(f"✅ API connection successful! BTC price: ${price:,.2f}")
                return True
            else:
                print("❌ Cannot get BTC price")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False

    def test_single_coin(self, coin_name: str = "ETH", days: int = 7) -> bool:
        """
        Test download single coin for several days

        Args:
            coin_name: Coin name
            days: Number of days

        Returns:
            bool: Whether test was successful
        """
        print(f"🧪 Testing download {coin_name} for last {days} days...")

        if coin_name not in self.coins:
            print(f"❌ Unsupported coin: {coin_name}")
            return False

        try:
            # Calculate test date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Get coin symbol
            symbol = self.coins[coin_name]['symbol']

            print(f"📥 Downloading {coin_name} ({symbol}) data...")
            print(f"📅 Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Download data
            raw_data = self.api.get_historical_data(
                symbol=symbol,
                interval=self.config['timeframe'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if not raw_data:
                print(f"❌ No data retrieved for {coin_name}")
                return False

            print(f"✅ Successfully downloaded {len(raw_data)} raw data points")

            # Convert data format
            standard_data = self.api.convert_to_standard_format(raw_data, coin_name)

            if not standard_data:
                print(f"❌ {coin_name} data conversion failed")
                return False

            print(f"✅ Successfully converted to standard format, total {len(standard_data)} records")

            # Validate data quality
            is_valid, message = self.api.validate_data_quality(standard_data)
            if is_valid:
                print(f"✅ Data quality validation passed: {message}")

                # Show data example
                if standard_data:
                    print(f"\n📊 Data example ({coin_name}):")
                    example = standard_data[0]
                    print(f"  Time: {datetime.fromtimestamp(example[0]).strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Open: {example[4]:.8f}")
                    print(f"  High: {example[2]:.8f}")
                    print(f"  Low: {example[3]:.8f}")
                    print(f"  Close: {example[5]:.8f}")
                    print(f"  Volume: {example[6]:.2f}")

                return True
            else:
                print(f"❌ Data quality validation failed: {message}")
                return False

        except Exception as e:
            print(f"❌ Error testing download {coin_name}: {e}")
            return False

    def download_all_coins(self, force_download: bool = False) -> bool:
        """
        Download complete data for all coins

        Args:
            force_download: Whether to force re-download

        Returns:
            bool: Whether download was successful
        """
        print("🚀 Starting to download all coins data...")

        # Check existing data
        status = self.manager.get_download_status()

        if status['exists'] and not force_download:
            print(f"📊 Database exists, current records: {status['total_records']:,}")
            print(f"🪙 Downloaded coins: {len(status['coins'])}/{len(self.coins)}")

            if status['missing_coins']:
                print(f"❌ Missing coins: {', '.join(status['missing_coins'])}")

            if status['incomplete_coins']:
                incomplete_info = ', '.join([f"{coin}({count:,})" for coin, count in status['incomplete_coins']])
                print(f"⚠️  Incomplete coins: {incomplete_info}")

            confirm = input("\nContinue to download all data? (Type 'YES' to confirm): ").strip()
            if confirm != 'YES':
                print("❌ Download cancelled")
                return False
        else:
            if force_download:
                print("⚠️  Force mode: will overwrite existing data")

            confirm = input(f"\nConfirm download all {len(self.coins)} coins data? (Type 'YES' to confirm): ").strip()
            if confirm != 'YES':
                print("❌ Download cancelled")
                return False

        print(f"\n🎯 Starting download...")
        start_time = datetime.now()

        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.manager.db_path), exist_ok=True)

        # Execute download
        success = self.manager.download_all_coins()

        # Show results
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n{'='*60}")
        if success:
            print("🎉 All coins download completed!")
        else:
            print("⚠️  Some coins download failed")
        print(f"⏱️  Total time: {duration}")

        # Show final statistics
        final_status = self.manager.get_download_status()
        print(f"📊 Final statistics:")
        print(f"  Total records: {final_status['total_records']:,}")
        print(f"  Completed coins: {len(final_status['coins'])}/{len(self.coins)}")
        print(f"  Database path: {self.manager.db_path}")

        return success

    def download_single_coin(self, coin_name: str, append_mode: bool = False) -> bool:
        """
        Download single coin data

        Args:
            coin_name: Coin name
            append_mode: Whether to use append mode

        Returns:
            bool: Whether download was successful
        """
        if coin_name not in self.coins:
            print(f"❌ Unsupported coin: {coin_name}")
            print(f"Supported coins: {', '.join(self.coins.keys())}")
            return False

        mode_text = "append mode" if append_mode else "complete mode"
        print(f"📥 Starting download {coin_name} ({mode_text})...")

        success = self.manager.download_coin(coin_name, append_mode=append_mode)

        if success:
            print(f"✅ {coin_name} download successful")

            # Show download statistics
            status = self.manager.get_download_status()
            if coin_name in status['coins']:
                coin_data = status['coins'][coin_name]
                print(f"  Records: {coin_data['count']:,}")
                print(f"  Time range: {datetime.fromtimestamp(coin_data['start_date']).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(coin_data['end_date']).strftime('%Y-%m-%d')}")
        else:
            print(f"❌ {coin_name} download failed")

        return success

    def show_download_status(self):
        """Display download status"""
        print("📊 Checking download status...")

        status = self.manager.get_download_status()

        if not status['exists']:
            print("❌ Database file does not exist")
            print(f"📁 Expected path: {self.manager.db_path}")
            return

        print(f"✅ Database file exists: {self.manager.db_path}")
        print(f"📈 Total records: {status['total_records']:,}")
        print(f"🪙 Downloaded coins: {len(status['coins'])}/{len(self.coins)}")

        if status['missing_coins']:
            print(f"❌ Missing coins: {', '.join(status['missing_coins'])}")

        if status['incomplete_coins']:
            incomplete_str = ', '.join([f"{coin}({count:,})" for coin, count in status['incomplete_coins']])
            print(f"⚠️  Incomplete coins: {incomplete_str}")

        if status['coins']:
            print(f"\n📋 Data volume by coin:")
            expected_count = 365 * 3 * 288  # Approximately 316,800 records for 3 years
            for coin, data in sorted(status['coins'].items(), key=lambda x: x[1]['count'], reverse=True):
                completeness = min((data['count'] / expected_count) * 100, 100)
                status_icon = "✅" if completeness >= 95 else "⚠️" if completeness >= 50 else "❌"
                print(f"  {status_icon} {coin:<12} {data['count']:>8,} records ({completeness:>5.1f}%)")

    def complete_missing_data(self) -> bool:
        """Complete missing data"""
        print("🔧 Starting to complete missing data...")

        status = self.manager.get_download_status()
        missing_coins = status['missing_coins']
        incomplete_coins = [coin for coin, _ in status['incomplete_coins']]

        if not missing_coins and not incomplete_coins:
            print("🎉 All data is complete, no need to supplement")
            return True

        print(f"Need to complete:")
        if missing_coins:
            print(f"  Missing coins: {', '.join(missing_coins)}")
        if incomplete_coins:
            print(f"  Incomplete coins: {', '.join(incomplete_coins)}")

        confirm = input("\nConfirm to start completion? (Type 'YES' to confirm): ").strip()
        if confirm != 'YES':
            print("❌ Completion cancelled")
            return False

        success = self.manager.complete_missing_data()

        if success:
            print("✅ Completion completed")

            # Show final status
            final_status = self.manager.get_download_status()
            print(f"📊 Final statistics:")
            print(f"  Total records: {final_status['total_records']:,}")
            print(f"  Completed coins: {len(final_status['coins'])}/{len(self.coins)}")
        else:
            print("❌ Error occurred during completion")

        return success

    def verify_database(self) -> bool:
        """Verify database integrity"""
        print("🔍 Verifying database integrity...")

        is_valid, message = self.manager.verify_database()

        if is_valid:
            print("✅ Database verification passed")

            # Show database information
            status = self.manager.get_download_status()
            if status['exists']:
                print(f"📊 Database information:")
                print(f"  Path: {self.manager.db_path}")
                print(f"  Records: {status['total_records']:,}")
                print(f"  Coins: {len(status['coins'])}")

                # Data quality check
                if status['total_records'] > 0:
                    expected_total = len(self.coins) * 365 * 3 * 288
                    completeness = min((status['total_records'] / expected_total) * 100, 100)
                    print(f"  Completeness: {completeness:.1f}%")
        else:
            print(f"❌ Database verification failed: {message}")

        return is_valid


def main():
    """Main function - Command line interface"""
    parser = argparse.ArgumentParser(
        description="PGPortfolio Binance Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s                                    # Interactive menu
  %(prog)s --test-api                        # Test API connection
  %(prog)s --test-coin ETH --days 7          # Test download ETH 7 days data
  %(prog)s --download-all                    # Download all coins
  %(prog)s --download BTC                    # Download single coin
  %(prog)s --download ETH --append           # Append download ETH
  %(prog)s --status                          # Check download status
  %(prog)s --complete                        # Complete missing data
  %(prog)s --verify                          # Verify database
        """
    )

    parser.add_argument(
        '--database', '-d',
        default='DataNew.db',
        help='Database file name (default: DataNew.db)'
    )

    parser.add_argument(
        '--test-api',
        action='store_true',
        help='Test Binance API connection'
    )

    parser.add_argument(
        '--test-coin',
        help='Test download single coin (default: ETH)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Test download days (default: 7)'
    )

    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all coins data'
    )

    parser.add_argument(
        '--download',
        help='Download single coin data'
    )

    parser.add_argument(
        '--append',
        action='store_true',
        help='Append mode (use with --download)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Check download status'
    )

    parser.add_argument(
        '--complete',
        action='store_true',
        help='Complete missing data'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify database integrity'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode, reduce output'
    )

    args = parser.parse_args()

    # Set log level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Initialize downloader
    try:
        downloader = BinanceDataDownloader(args.database)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

    # Execute corresponding operation based on parameters
    try:
        if args.test_api:
            success = downloader.test_api_connection()
            sys.exit(0 if success else 1)

        elif args.test_coin:
            success = downloader.test_single_coin(args.test_coin, args.days)
            sys.exit(0 if success else 1)

        elif args.download_all:
            downloader.print_welcome()
            success = downloader.download_all_coins(force_download=args.force)
            sys.exit(0 if success else 1)

        elif args.download:
            success = downloader.download_single_coin(args.download, append_mode=args.append)
            sys.exit(0 if success else 1)

        elif args.status:
            downloader.show_download_status()

        elif args.complete:
            downloader.complete_missing_data()

        elif args.verify:
            success = downloader.verify_database()
            sys.exit(0 if success else 1)

        else:
            # Interactive menu
            interactive_menu(downloader)

    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing operation: {e}")
        sys.exit(1)


def interactive_menu(downloader: BinanceDataDownloader):
    """Interactive menu"""
    while True:
        try:
            print("\n" + "="*60)
            print("🚀 PGPortfolio Binance Data Downloader")
            print("="*60)
            print("1. 🧪 Test API connection")
            print("2. 🧪 Test download single coin")
            print("3. 📥 Download all coins data")
            print("4. 📥 Download single coin")
            print("5. 📊 Check download status")
            print("6. 🔧 Complete missing data")
            print("7. 🔍 Verify database integrity")
            print("8. ❌ Exit")
            print("="*60)

            choice = input("Please select operation (1-8): ").strip()

            if choice == '1':
                downloader.test_api_connection()

            elif choice == '2':
                coin = input("Coin name (default: ETH): ").strip() or "ETH"
                try:
                    days = int(input("Test days (default: 7): ").strip() or "7")
                except ValueError:
                    days = 7

                downloader.test_single_coin(coin, days)

            elif choice == '3':
                downloader.download_all_coins()

            elif choice == '4':
                coin = input("Coin name: ").strip()
                if coin:
                    append = input("Append mode? (y/N): ").strip().lower() == 'y'
                    downloader.download_single_coin(coin, append_mode=append)

            elif choice == '5':
                downloader.show_download_status()

            elif choice == '6':
                downloader.complete_missing_data()

            elif choice == '7':
                downloader.verify_database()

            elif choice == '8':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid selection, please try again")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Operation error: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()