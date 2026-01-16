"""
Configuration file for Machine Translation project
Optimized for NVIDIA GTX 1660 SUPER (6GB VRAM)
"""

import torch

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if USE_CUDA else "CPU"

# For GTX 1660 SUPER optimization
PIN_MEMORY = True if USE_CUDA else False
NUM_WORKERS = 4  # Adjust based on your CPU cores

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_DIR = 'data'
DATASET_URL = 'https://www.manythings.org/anki/fra-eng.zip'
DATASET_FILE = 'fra.txt'

# Preprocessing
MAX_LENGTH = 15
MIN_FREQ = 2
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Seq2Seq Configuration
SEQ2SEQ_CONFIG = {
    'embedding_dim': 256,
    'hidden_dim': 512,
    'dropout': 0.1,
    'teacher_forcing_ratio': 0.5
}

# Transformer Configuration
TRANSFORMER_CONFIG = {
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'max_len': 5000
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Optimized for GTX 1660 SUPER (6GB VRAM)
BATCH_SIZE = 64  # Reduce to 32 if OOM
EPOCHS = 10
LEARNING_RATE_SEQ2SEQ = 0.001
LEARNING_RATE_TRANSFORMER = 0.0001

# Gradient clipping
CLIP_GRAD = 1.0

# Early stopping
PATIENCE = 3
MIN_DELTA = 0.001

# Checkpointing
SAVE_DIR = 'models'
CHECKPOINT_FREQ = 1  # Save every N epochs

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
BEAM_WIDTH = 3
MAX_DECODE_LENGTH = 20

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
BLEU_SAMPLES = 1000  # Number of samples for BLEU calculation
ERROR_ANALYSIS_SAMPLES = 200

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_DIR = 'logs'
OUTPUT_DIR = 'outputs'
LOG_INTERVAL = 100  # Log every N batches

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
PLOT_DPI = 300
FIGURE_SIZE = (12, 6)

# ============================================================================
# RANDOM SEEDS
# ============================================================================
SEED = 42

# ============================================================================
# GPU MONITORING
# ============================================================================
def print_gpu_info():
    """Print GPU information"""
    if USE_CUDA:
        print("="*80)
        print("GPU CONFIGURATION")
        print("="*80)
        print(f"Device: {CUDA_DEVICE_NAME}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"\nMemory Info:")
        print(f"  Total: {total_memory:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Available: {total_memory - allocated:.2f} GB")
        print("="*80)
    else:
        print("WARNING: CUDA not available. Running on CPU.")
        print("Training will be significantly slower.")

if __name__ == '__main__':
    print_gpu_info()
