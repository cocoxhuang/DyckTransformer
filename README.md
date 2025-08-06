# DyckTransformer

A PyTorch implementation of a Transformer model for learning Dyck language patterns. This project focuses on training encoder-decoder transformers to understand and generate balanced parentheses sequences (Dyck words) with comprehensive attention mechanism analysis and robust training infrastructure.

## Features

- **Transformer Architecture**: Supports encoder-only, decoder-only, and encoder-decoder configurations
- **Dyck Language Generation**: Automated generation of balanced parentheses sequences using SageMath
- **Attention Visualization**: Interactive analysis tools for cross-attention and self-attention patterns
- **Jupyter Notebook Analysis**: Ready-to-use notebooks for model analysis and visualization

## 📁 Project Structure

```
DyckTransformer/
├── src/
│   ├── model/
│   │   └── transformer.py          # Transformer architecture implementation
│   ├── data/
│   │   ├── dataset.py              # Dataset handling and preprocessing
│   │   └── dyck_generator.py       # Dyck words generation with caching
│   ├── training/
│   │   ├── trainer.py              # Training orchestration class with checkpointing
│   │   └── evaluator.py            # Model evaluation utilities
│   ├── utils/
│   │   ├── config.py               # Configuration management
│   │   └── logger.py               # Logging system with session management
│   └── config/
│       └── default_config.py       # Default configuration settings
├── configs/
│   └── default_config.yaml         # YAML configuration file
├── cache/                           # Cached data and model checkpoints
│   └── sesh_YYYYMMDD_HHMMSS/       # Timestamped training sessions
├── examples/
│   └── resume_training.py          # Example scripts for training resumption
├── Analysis.ipynb        # Analysis notebook with attention visualization
├── train.py                        # Main training script with resume support
├── manage_sessions.py              # Session management utility
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- SageMath (for Dyck words generation)
- CUDA-compatible GPU (optional, for faster training)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/cocoxhuang/DyckTransformer.git
cd DyckTransformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install SageMath (if not already installed):
```bash
# On Ubuntu/Debian
sudo apt-get install sagemath

# On macOS with Homebrew
brew install sagemath

# Or use conda
conda install -c conda-forge sage
```

## Quick Start

## Quick Start

### Basic Training
Start a new training session:
```bash
python train.py
```

With custom configuration:
```bash
python train.py --config configs/custom_config.yaml
```

### Training Resumption
Resume training from a previous session:
```bash
python train.py --resume cache/sesh_20250806_123456
```

## Configuration

The project uses YAML configuration files for easy parameter management. Key configuration sections:

```yaml
# Model architecture settings
model:
  architecture: "encoder_decoder"
  d_model: 128
  num_heads: 4
  num_encoder_layers: 1
  num_decoder_layers: 1
  dropout: 0.0

# Training parameters
training:
  batch_size: 32
  num_epochs: 150     
  learning_rate: 0.0001
  weight_decay: 0.0
  cache_dir: "cache"
  seed: 42
  save_every_epoch: true                # Save model after each epoch

# Data generation settings
data:
  n: 13                                 # Dyck words semilength
  data_path: 'data/dyck_data_13.pkl'   # Dataset cache path
  force_regenerate: false               # Force data regeneration
  eval_size: 10000                      # Validation set size
```

## Configuration

The project uses YAML configuration files for easy parameter management. Key configuration sections:

- **data**: Dataset parameters (n, data_path, cache_dir)
- **model**: Architecture settings (d_model, nhead, num_layers, etc.)
- **training**: Training parameters (batch_size, learning_rate, epochs, etc.)

Example configuration:
```yaml
data:
  n: 10
  data_path: "cache/dyck_data.pkl"
  cache_dir: "cache"

model:
  architecture: "encoder_decoder"
  d_model: 128
  nhead: 8
  num_layers: 6

training:
  batch_size: 32
  num_epochs: 150     
  learning_rate: 0.0001
  weight_decay: 0.0
  cache_dir: "cache"
  seed: 42
  save_every_epoch: true                            # Save model after each epoch
```

## Analysis Tools

The project provides comprehensive analysis capabilities through interactive Jupyter notebooks and programmatic functions.

### **Interactive Analysis Functions**

#### Cross-Attention Analysis
```python
# Analyze multiple examples in a grid layout
outputs = analyze_cross_attention(
    model, examples, 
    start_idx=0,        # Starting example index
    step=15,            # Generation step to analyze  
    att_head=2,         # Attention head to visualize
    num_examples=9,     # Number of examples (3x3 grid)
    show_diagonal=False # Optional diagonal reference line
)
```

#### Self-Attention Analysis  
```python
# Analyze individual example self-attention patterns
outputs = analyze_self_attention(
    model, examples,
    ex_idx=10,          # Example index
    step=21,            # Generation step
    att_head=1,         # Attention head
    cmap='Blues'        # Color scheme
)
```

### Initial Embedding Analysis
See Analysis.ipynb
