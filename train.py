import argparse
from src.model.transformer import Transformer
from src.data.dataset import Dataset
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config
from src.data.dyck_generator import generate_dyck_data
from src.utils.logger import Logger
import os

def main(config_path, resume_from=None):
    config = load_config(config_path)
    
    if resume_from:
        # Load configuration from the resume session
        resume_config_path = os.path.join(resume_from, "config")
        if os.path.exists(resume_config_path):
            config = load_config(resume_config_path)
            # Use the old configuration but allow some overrides from new config
            logger = Logger(resume_from=resume_from)
            logger.info(f"Resuming training from session: {resume_from}")
        else:
            logger = Logger(cache_dir=config['training']['cache_dir'])
            logger.error(f"Resume session config not found: {resume_config_path}")
            return
    else:
        # Create new logger for fresh training
        cache_dir = config['training']['cache_dir']
        logger = Logger(cache_dir=cache_dir)
    
    # Generate Dyck words data with caching
    n = config['data']['n']
    data_path = config['data'].get('data_path', None)  # None will use default path with n
    force_regenerate = config['data'].get('force_regenerate', False)
    data = generate_dyck_data(n, data_path=data_path, force_regenerate=force_regenerate, logger=logger)
    # Initialize the dataset
    dataset = Dataset(data, config['data']['seed'], batch_size=config['training']['batch_size'])

    # Update max_len based on actual data
    target_len = dataset.train_dataloader.dataset[1][1].shape[0]    # assuming they are all the same len
    input_len = dataset.train_dataloader.dataset[1][0].shape[0]
    if config['model']['architecture'] == 'decoder_only':
        # For decoder-only, we concatenate input + target, so need longer max_len
        max_len = input_len + target_len
    else:
        # For other architectures, use target length or max of input/target
        max_len = max(input_len, target_len)

    # Initialize the model
    model = Transformer(
        src_vocab_size=config['model']['src_vocab_size'],
        tgt_vocab_size=config['model']['tgt_vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        max_len=max_len,  # Use actual data length
        dropout=config['model']['dropout'],
        architecture=config['model']['architecture'],
        is_sinusoidal=config['model']['is_sinusoidal']
    )

    # Initialize the trainer (it will create its own logger)
    trainer = Trainer(model, dataset, config, logger, resume_from=resume_from)

    # Start training
    results = trainer.train()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Transformer model.")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to training session to resume from (e.g., cache/sesh_20250804_052923)')
    # parser.add_argument('--list-sessions', action='store_true', help='List all available training sessions')
    args = parser.parse_args()

    main(args.config, resume_from=args.resume)