"""
config.py — Global constants and seed initialization
P08: Stock Market Sentiment and Price Movement Predictor
"""

import random
import numpy as np
import torch

# ──────────────────────────────────────────────
# Reproducibility — set random seeds
# ──────────────────────────────────────────────
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────
# Stock & Date Configuration
# ──────────────────────────────────────────────
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────
LOOKBACK = 30          # Number of past days used as input sequence
HIDDEN_SIZE = 128      # LSTM hidden state size
NUM_LAYERS = 2         # Number of stacked LSTM layers
DROPOUT = 0.2          # Dropout rate for regularisation
BATCH_SIZE = 32        # Training batch size
EPOCHS = 30            # Number of training epochs

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
