import unittest
import pandas as pd
from datetime import datetime, timedelta

from tests.integration.setup_integration_tests import logger
from config.system_config import SystemConfig
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_processors.data_pipeline import DataPipeline
from environments.trading_gym.trading_env import TradingEnv
from models.agents.dqn_agent import DQNAgent
from backtesting.backtester import Backtester
from deployment.trade_executor import TradeExecutor

class TestEndToEndFlow(unittest.TestCase):
    """Kiểm thử luồng hoạt động từ đầu đến cuối của hệ thống"""
    
    @classmethod
    def setUpClass(cls):
        """Khởi tạo các thành phần cần thiết cho kiểm thử"""
        # Khởi tạo cấu hình hệ thống
        cls.config = SystemConfig()
        cls.config.load_from_file('config/test_config.yaml')
        
        # Thông số kiểm thử
        cls.symbol = "BTC/USDT"
        cls.timeframe = "1h"
        cls.start_date = datetime.now() - timedelta(days=30)
        cls.end_date = datetime.now()
        
        logger.info(f"Setting up end-to-end test with {cls.symbol} {cls.timeframe}")
    
    def test_data_collection_and_processing(self):
        """Kiểm thử thu thập và xử lý dữ liệu"""
        # Thu thập dữ liệu
        connector = BinanceConnector(api_key="test", api_secret="test")
        data = connector.get_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
        # Xử lý dữ liệu
        pipeline = DataPipeline(self.config)
        processed_data = pipeline.process(data)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertGreater(len(processed_data.columns), len(data.columns))
        
        logger.info(f"Data collection and processing successful. Generated {len(processed_data.columns)} features.")
    
    def test_agent_training_and_evaluation(self):
        """Kiểm thử huấn luyện và đánh giá agent"""
        # Tạo môi trường huấn luyện
        env = TradingEnv(
            data=self.get_test_data(),
            initial_balance=10000,
            commission=0.001
        )
        
        # Khởi tạo và huấn luyện agent
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config=self.config
        )
        
        # Huấn luyện nhanh để kiểm thử
        agent.train(env, episodes=5, batch_size=32)
        
        # Đánh giá agent
        backtester = Backtester(env, agent)
        results = backtester.run()
        
        self.assertIsInstance(results, dict)
        self.assertIn('total_profit', results)
        
        logger.info(f"Agent training and evaluation successful. Profit: {results['total_profit']}")
    
    def test_trade_execution(self):
        """Kiểm thử thực thi giao dịch"""
        # Khởi tạo executor với cấu hình test
        executor = TradeExecutor(
            agent=self.get_trained_agent(),
            exchange_connector=BinanceConnector(api_key="test", api_secret="test"),
            config=self.config,
            test_mode=True
        )
        
        # Thực thi một lệnh giao dịch
        order_result = executor.execute_trade(
            symbol=self.symbol,
            action="buy",
            amount=0.01
        )
        
        self.assertIsInstance(order_result, dict)
        self.assertIn('status', order_result)
        
        logger.info(f"Trade execution test successful.")
    
    def get_test_data(self):
        """Helper để lấy dữ liệu kiểm thử"""
        # Đây chỉ là giả lập, trong thực tế bạn sẽ lấy dữ liệu thật
        return pd.DataFrame({
            'open': [40000, 41000, 42000],
            'high': [41000, 42000, 43000],
            'low': [39000, 40000, 41000],
            'close': [41000, 42000, 42500],
            'volume': [1000, 1200, 1100]
        })
    
    def get_trained_agent(self):
        """Helper để lấy agent đã huấn luyện"""
        # Trong thực tế bạn sẽ load model đã huấn luyện
        env = TradingEnv(
            data=self.get_test_data(),
            initial_balance=10000,
            commission=0.001
        )
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config=self.config
        )
        return agent

if __name__ == '__main__':
    unittest.main()