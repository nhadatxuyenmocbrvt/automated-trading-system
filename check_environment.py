# check_environment.py
import os
import sys

print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not found")

try:
    import gym
    print(f"Gym version: {gym.__version__}")
except ImportError:
    print("Gym not found")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("Pandas not found")

# Kiểm tra các module tự phát triển
modules_to_check = [
    "models.agents.dqn_agent",
    "models.networks.value_network",
    "environments.trading_gym.trading_env",
    "models.training_pipeline.trainer"
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f"Successfully imported {module}")
    except ImportError as e:
        print(f"Failed to import {module}: {str(e)}")