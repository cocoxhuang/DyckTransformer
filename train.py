import argparse
import torch
from src.model.transformer import Transformer
from src.data.dataset import Dataset
from src.training.trainer import Trainer
from src.utils.config import load_config

def main(config_path):
    config = load_config(config_path)

    # Initialize the dataset
    dataset = Dataset(config['data'], batch_size=config['training']['batch_size'])

    # Initialize the model
    model = Transformer(
        src_vocab_size=config['model']['src_vocab_size'],
        tgt_vocab_size=config['model']['tgt_vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        max_len=config['model']['max_len'],
        dropout=config['model']['dropout'],
        architecture=config['model']['architecture'],
        is_sinusoidal=config['model']['is_sinusoidal']
    )

    # Initialize the trainer
    trainer = Trainer(model, dataset, config)

    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    main(args.config)