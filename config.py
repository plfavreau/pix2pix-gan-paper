import torch

# Training hyperparameters
BATCH_SIZE = 8
WORKERS = 10
EPOCHS = 100
L1_LAMBDA = 100.0
LEARNING_RATE = 2e-4
BETA1 = 0.5
BETA2 = 0.999

# Model dimensions
GF_DIM = 64
DF_DIM = 64
C_DIM = 3

# Data paths
TRAIN_PATH = "facades/train/"
VAL_PATH = "facades/val/"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint and results paths
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
