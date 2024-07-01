import torch

# General settings
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data settings
BATCH_SIZE = 128
JUST_NORMAL = True
NORMAL_CLASS = 0
AUGMENTATION = True
MODE = "acrosome"

# Model settings
USE_BIAS = False
BATCH_NORM = True
TARGET_LAYER = 11
CFG = [16, "M", 16, 128, "M", 256, "M", 128, 512, "M", 32, 512, "M"]

# Training settings
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
SCHEDULER = False
FGSM_ATTACK_ENABLE = True
EPSILON = 0.04
CRIT = 4
LAMBDA = 0.01

# VGG and Custom VGG important layers
VGG_IMPORTANT_LAYERS = [9, 16, 23, 30]
MODEL_IMPORTANT_LAYERS = [8, 11, 17, 23]
