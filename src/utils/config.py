from pathlib import Path
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def get_default_config():
    return {
        'model': {
            'src_vocab_size': 4,
            'tgt_vocab_size': 4,
            'd_model': 128,
            'num_heads': 4,
            'd_ff': 256,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'max_len': 128,
            'dropout': 0.1,
            'architecture': 'encoder_decoder',
            'is_sinusoidal': False
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.0001,
            'weight_decay': 0
        },
        'data': {
            'train_data_path': 'path/to/train/data',
            'eval_data_path': 'path/to/eval/data'
        }
    }