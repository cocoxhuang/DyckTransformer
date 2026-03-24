# DyckTransformer

A PyTorch implementation of a Transformer trained to learn the `area_dinv_to_bounce_area` bijection on Dyck words. Given a Dyck word as input, the model predicts its image under this map. The project includes flexible architecture configuration, session-based checkpointing, and attention visualization tools.

## Task

The model is trained on all Dyck words of a given semilength `n`. Inputs and targets are token sequences (0/1 steps) encoding lattice paths, generated via SageMath's `DyckWords`. Token ids 0 and 1 are reserved for BOS/EOS; Dyck step values are shifted by +2.

## Project Structure

```
DyckTransformer/
├── src/
│   ├── model/
│   │   └── transformer.py          # Transformer architecture
│   ├── data/
│   │   ├── dataset.py              # Dataset wrapper and dataloader creation
│   │   └── dyck_generator.py       # Dyck word generation and caching (requires SageMath)
│   ├── training/
│   │   ├── trainer.py              # Training loop, optimizer, checkpointing
│   │   └── evaluator.py            # Loss and sequence-level accuracy evaluation
│   └── utils/
│       ├── config.py               # YAML config loading and saving
│       ├── logger.py               # Session-based logging
│       ├── transformer_analysis.py # Attention visualization functions
│       └── utils.py                # Dyck word lattice path utilities
├── configs/
│   └── default_config.yaml         # Default configuration
├── cache/                          # Training sessions (logs, checkpoints, configs)
│   └── sesh_YYYYMMDD_HHMMSS/
├── data/                           # Cached dataset files
├── Analysis.ipynb                  # Attention and embedding analysis notebook
├── algo.ipynb                      # Algorithm exploration notebook
├── area_dinv_eda.ipynb             # Area/dinv EDA notebook
├── train.py                        # Main training script
├── dyck_data.py                    # Standalone data generation script
└── requirements.txt
```

## Installation

**Prerequisites:**
- Python 3.8+
- SageMath (for Dyck word generation)
- CUDA-compatible GPU (optional)

```bash
git clone https://github.com/cocoxhuang/DyckTransformer.git
cd DyckTransformer
pip install -r requirements.txt
```

SageMath installation:
```bash
# Ubuntu/Debian
sudo apt-get install sagemath

# macOS
brew install sagemath

# conda
conda install -c conda-forge sage
```

## Data Generation

Generate and cache a dataset for semilength `n` independently of training:
```bash
python dyck_data.py --n 13
```

This saves to `data/dyck_data_13.pkl`. Training will also generate and cache data automatically on first run.

## Training

Start training with the default config:
```bash
python train.py
```

With a custom config:
```bash
python train.py --config configs/default_config.yaml
```

Resume a previous session:
```bash
python train.py --resume cache/sesh_20250806_123456
```

Each training session is saved under `cache/sesh_YYYYMMDD_HHMMSS/` and includes the config, logs, and per-epoch model checkpoints.

## Configuration

Edit `configs/default_config.yaml` to control all parameters:

```yaml
model:
  architecture: 'encoder_decoder'  # 'encoder_only', 'decoder_only', or 'encoder_decoder'
  d_model: 128
  num_heads: 4
  d_ff: 256
  num_encoder_layers: 1
  num_decoder_layers: 1
  dropout: 0
  is_sinusoidal: false             # false = learnable positional encoding
  seed: 5

training:
  batch_size: 32
  num_epochs: 33
  learning_rate: 0.0001
  weight_decay: 0.0
  cache_dir: 'cache'
  seed: 5

data:
  n: 13                            # Dyck word semilength
  data_path: 'data/dyck_data_13.pkl'
  eval_size: 10000
  force_regenerate: false
  seed: 42
```

## Architecture

The `Transformer` class in [src/model/transformer.py](src/model/transformer.py) supports three modes set by `architecture`:

- `encoder_decoder`: Standard encoder-decoder with cross-attention (default)
- `encoder_only`: Encoder stack with a linear output head
- `decoder_only`: Causal decoder stack; input and target are concatenated

All architectures use post-norm (LayerNorm after residual), multi-head scaled dot-product attention, and GELU feedforward blocks. Positional encoding is learnable by default; sinusoidal is available via `is_sinusoidal: true`.

Generation supports greedy decoding (`deterministic=True`) and sampling with temperature and top-k.

## Evaluation

The `Evaluator` computes:
- **Loss**: cross-entropy over the target sequence
- **Accuracy**: sequence-level exact match over the generated tokens

## Analysis

[Analysis.ipynb](Analysis.ipynb) provides interactive analysis using functions from `src/utils/transformer_analysis.py`:

| Function | Description |
|---|---|
| `analyze_cross_attention` | Cross-attention heatmaps for multiple examples at a given generation step |
| `analyze_decoder_attention` | Decoder self-attention heatmaps for multiple examples |
| `analyze_encoder_attention` | Encoder self-attention heatmaps for multiple examples |
| `analyze_cross_attention_all_steps` | Cross-attention across all generation steps for one example, with Dyck path plots |

All functions accept `att_head`, `step`, `num_examples`, `cmap`, and `show_diagonal` parameters and return the collected attention tensors.

Embedding weights are accessible via `model._embedding_weights()`. The `word_to_path` utility converts token sequences to (x, y) lattice path coordinates for plotting.
