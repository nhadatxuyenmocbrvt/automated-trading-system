import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from tests.integration.setup_integration_tests import logger
from config.system_config import SystemConfig
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_processors.feature_engineering.technical_indicators.trend_indicators import TrendIndicators
from environments.trading_gym.trading_env import TradingEnv
from environments.simulators.exchange_simulator.realistic_simulator import RealisticExchangeSimulator
from models.agents.dqn_agent import DQNAgent
from backtesting.evaluation.strategy_evaluator import StrategyEvaluator
from risk_management.portfolio_manager import PortfolioManager
from deployment.exchange_api.order_manager import OrderManager
from agent_manager.agent_coordinator import AgentCoordinator
from real_time_inference.inference_engine import InferenceEngine

class TestModuleIntegration(unittest.TestCase):
    """Kiểm thử tích hợp giữa các module chính"""
    
    @classmethod
    def setUpClass(cls):
        """Khởi tạo các thành phần cần thiết cho kiểm thử"""
        cls.config = SystemConfig()
        cls.config.load_from_file('config/test_config.yaml')
        cls.symbol = "BTC/USDT"
        
        # Tạo dữ liệu test cơ bản
        cls.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'open': np.random.normal(40000, 1000, 100),
            'high': np.random.normal(41000, 1000, 100),
            'low': np.random.normal(39000, 1000, 100),
            'close': np.random.normal(40500, 1000, 100),
            'volume': np.random.normal(1000, 200, 100)
        })
        cls.sample_data.set_index('timestamp', inplace=True)
        
        logger.info("Setting up module integration tests")
    
    def test_data_to_features_integration(self):
        """Kiểm thử tích hợp từ dữ liệu thô đến tạo đặc trưng"""
        # Tạo indicators
        trend_indicators = TrendIndicators()
        
        # Áp dụng indicators
        data_with_features = trend_indicators.add_indicators(self.sample_data.copy())
        
        # Kiểm tra các đặc trưng đã được tạo
        self.assertIn('sma_20', data_with_features.columns)
        self.assertIn('ema_14', data_with_features.columns)
        
        logger.info(f"Data to features integration successful. Generated indicators: {data_with_features.columns.tolist()}")
    
    def test_environment_to_agent_integration(self):
        """Kiểm thử tích hợp từ môi trường đến agent"""
        # Tạo môi trường huấn luyện
        env = TradingEnv(
            data=self.sample_data,
            initial_balance=10000,
            commission=0.001
        )
        
        # Khởi tạo agent
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config=self.config
        )
        
        # Kiểm tra tương tác agent với môi trường
        state = env.reset()
        action = agent.act(state, explore=True)
        next_state, reward, done, info = env.step(action)
        
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(action, int)
        self.assertIsInstance(reward, float)
        
        logger.info(f"Environment to agent integration successful. Action: {action}, Reward: {reward}")
    
    def test_agent_to_evaluation_integration(self):
        """Kiểm thử tích hợp từ agent đến đánh giá"""
        # Khởi tạo môi trường và agent (đơn giản hóa)
        env = TradingEnv(
            data=self.sample_data,
            initial_balance=10000,
            commission=0.001
        )
        agent = DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config=self.config
        )
        
        # Đánh giá chiến lược
        evaluator = StrategyEvaluator()
        metrics = evaluator.evaluate_agent(agent, env)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        logger.info(f"Agent to evaluation integration successful. Metrics: {metrics}")
    
    def test_risk_management_integration(self):
        """Kiểm thử tích hợp quản lý rủi ro"""
        # Khởi tạo portfolio manager
        portfolio_manager = PortfolioManager(
            initial_balance=10000,
            risk_profile='moderate'
        )
        
        # Thêm vị thế
        portfolio_manager.add_position(
            symbol=self.symbol,
            amount=0.1,
            entry_price=40000
        )
        
        # Kiểm tra sức khỏe danh mục
        portfolio_health = portfolio_manager.check_portfolio_health()
        position_risk = portfolio_manager.calculate_position_risk(self.symbol)
        
        self.assertIsInstance(portfolio_health, dict)
        self.assertIsInstance(position_risk, float)
        
        logger.info(f"Risk management integration successful. Portfolio health: {portfolio_health}")
    
    def test_deployment_integration(self):
        """Kiểm thử tích hợp triển khai"""
        # Khởi tạo order manager (đặt chế độ test)
        order_manager = OrderManager(
            exchange_connector=BinanceConnector(api_key="test", api_secret="test"),
            test_mode=True
        )
        
        # Thực thi lệnh test
        order_result = order_manager.place_limit_order(
            symbol=self.symbol,
            side="buy",
            amount=0.01,
            price=40000
        )
        
        self.assertIsInstance(order_result, dict)
        
        logger.info(f"Deployment integration successful. Order result: {order_result}")
    
    def test_advanced_features_integration(self):
        """Kiểm thử tích hợp tính năng nâng cao"""
        # Khởi tạo agent coordinator
        coordinator = AgentCoordinator(config=self.config)
        
        # Khởi tạo inference engine
        inference_engine = InferenceEngine(
            agent_coordinator=coordinator,
            config=self.config
        )
        
        # Thực hiện inference
        action = inference_engine.get_action(
            symbol=self.symbol,
            market_data=self.sample_data.iloc[-10:],
            test_mode=True
        )
        
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        self.assertIn('confidence', action)
        
        logger.info(f"Advanced features integration successful. Action: {action}")

if __name__ == '__main__':
    unittest.main()