import unittest
import subprocess
import logging

class SystemIntegrationTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(filename='logs/integration_test.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('🚀 Starting System Integration Test Suite')

    def test_collect_data(self):
        result = subprocess.run(['python', 'main.py', 'collect', '--symbol', 'BTCUSDT'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        logging.info('✅ Data Collection Test Passed')

    def test_process_data(self):
        result = subprocess.run(['python', 'main.py', 'process', '--symbol', 'BTCUSDT'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        logging.info('✅ Data Processing Test Passed')

    def test_train_agent(self):
        result = subprocess.run(['python', 'main.py', 'train', '--agent', 'DQN'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        logging.info('✅ Agent Training Test Passed')

    def test_trade(self):
        result = subprocess.run(['python', 'main.py', 'trade', '--symbol', 'BTCUSDT'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        logging.info('✅ Trading Test Passed')

    def test_backtest(self):
        result = subprocess.run(['python', 'main.py', 'backtest', '--symbol', 'BTCUSDT'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        logging.info('✅ Backtest Test Passed')

    def test_dashboard(self):
        result = subprocess.run(['python', 'main.py', 'dashboard'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        logging.info('✅ Dashboard Test Passed')

if __name__ == '__main__':
    unittest.main()

