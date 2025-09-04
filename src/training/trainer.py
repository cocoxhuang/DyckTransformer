from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import random

# Import configuration
from ..utils.config import load_config, save_config
from ..utils.logger import Logger
from ..data.dataset import Dataset
from ..model.transformer import Transformer
from .evaluator import Evaluator
from ..data.dyck_generator import generate_dyck_data

class Trainer:
    def __init__(self, model, dataset, config, logger, resume_from=None):
        """
        Initialize the Trainer.
        
        Args:
            model: The transformer model to train
            dataset: The dataset object containing train/eval dataloaders
            config: Configuration dictionary
            logger: Logger instance
            resume_from: Path to training session to resume from (optional)
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger

        # set random seeds for reproducibility
        seed = config['training']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For CUDA reproducibility
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Setup logging
        cache_dir = config['training'].get('cache_dir', 'cache')
        
        # Extract training configuration
        self.num_epochs = config['training']['num_epochs']
        self.lr = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Training using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize evaluator
        self.evaluator = Evaluator(model, dataset.eval_dataloader, self.criterion, self.device)
        
        # Training tracking
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # Calculate max_new_tokens from dataset
        self.max_new_tokens = dataset.train_dataloader.dataset[1][1].shape[0] - 1  # Exclude BOS token
        
        # Handle training resumption
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self._print_model_info()
    
    def _print_model_info(self):
        """Print and log model information."""
        self.logger.info("============ Model Information ============")
        arch_msg = f"Model is {'decoder only' if self.model.architecture == 'decoder_only' else 'encoder-decoder' if self.model.architecture == 'encoder_decoder' else 'encoder only'}."
        self.logger.info(arch_msg)
        self.logger.info(f"Training using device: {self.device}")
        self.logger.info(f"Model has {self.config['model']['num_encoder_layers']} encoder layers, {self.config['model']['num_decoder_layers']} decoder layers, embedding dimension is {self.config['model']['d_model']}.")
        self.logger.info(f"Model has {self.config['model']['num_heads']} attention heads.")
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {param_count:,}")
        pos_enc_msg = f"Model is using {'sinusoidal' if self.model.is_sinusoidal else 'learnable'} positional encoding."
        self.logger.info(pos_enc_msg)
        
        # Log configuration
        self.logger.debug(f"Training configuration: {self.config}")
        config_path = self.logger.get_config_path()
        save_config(self.config, config_path)
        self.logger.info(f"Configuration saved to {config_path}")

    def _load_checkpoint(self, session_path):
        """Load training checkpoint from a previous session."""
        try:
            # Load model state
            model_path = os.path.join(session_path, "model.pth")
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Model loaded from {model_path}")
            
            # Load training state
            state_path = os.path.join(session_path, "training_state.pth")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location=self.device)
                self.start_epoch = state['epoch'] + 1
                self.train_loss = state['train_loss']
                self.eval_loss = state['eval_loss']
                self.eval_accuracy = state['eval_accuracy']
                self.best_loss = state['best_loss']
                self.optimizer.load_state_dict(state['optimizer_state'])
                self.logger.info(f"Training state loaded from {state_path}")
                self.logger.info(f"Resuming from epoch {self.start_epoch}")
            else:
                self.logger.warning(f"Training state file not found: {state_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            self.logger.info("Starting training from scratch")
            self.start_epoch = 0

    def _save_checkpoint(self, epoch, is_best=False):
        """Save training checkpoint."""
        try:
            # Save current model state
            model_path = self.logger.get_model_path("model")
            torch.save(self.model.state_dict(), model_path)
            
            # Save training state
            state = {
                'epoch': epoch,
                'train_loss': self.train_loss,
                'eval_loss': self.eval_loss,
                'eval_accuracy': self.eval_accuracy,
                'best_loss': self.best_loss,
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config
            }
            state_path = os.path.join(self.logger.cache_dir, "training_state.pth")
            torch.save(state, state_path)
            
            # Save epoch-specific checkpoint if enabled
            epoch_model_path = self.logger.get_model_path("model")
            torch.save(self.model.state_dict(), epoch_model_path)
            self.logger.debug(f"Epoch {epoch+1} model saved to {epoch_model_path}")
            
            # Save best model separately
            if is_best:
                best_model_path = self.logger.get_model_path("best_model")
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"Best model saved to {best_model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def train(self):        
        self.logger.info("============ Start Training ============")
        if self.start_epoch > 0:
            self.logger.info(f"Resuming training from epoch {self.start_epoch}")
        training_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(self.dataset.train_dataloader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                if self.model.architecture == 'encoder_decoder':
                    outputs = self.model(
                        src=inputs,
                        tgt=targets[:, :-1]  # exclude last token
                    )
                    # Calculate loss on shifted output
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        targets[:, 1:].contiguous().view(-1)  # exclude first token
                    )
                elif self.model.architecture == 'encoder_only':
                    outputs = self.model(
                        src=inputs
                    )
                    # Calculate loss on shmaifted output
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        targets.view(-1)
                    )
                else:  # decoder_only
                    # For decoder-only: concatenate input and target sequences
                    full_sequence = torch.cat([inputs, targets], dim=1)  # [batch_size, input_len + target_len]
                    outputs = self.model(tgt=full_sequence[:, :-1])  # Exclude last token for input. Trying to predict full_sequence[:, 1:] 
                    
                    # Calculate loss on shifted output (predict next token)
                    # We only want to calculate loss on the target portion
                    input_len = inputs.size(1)
                    pred_targets = outputs[:, input_len-1:]  # Start from the first target toke
                    loss = self.criterion(
                        pred_targets.contiguous().view(-1, pred_targets.size(-1)),
                        targets.view(-1)
                    )
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            # Average loss over batches
            avg_train_loss = total_loss / len(self.dataset.train_dataloader)
            self.train_loss.append(avg_train_loss)
            epoch_time = time.time() - epoch_start_time
            train_msg = f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s"
            self.logger.info(train_msg)

            # Evaluation
            with torch.no_grad():
                self.model.eval()
                eval_start_time = time.time()
                
                eval_loss_epoch, eval_accuracy_epoch = self.evaluator.evaluate(self.max_new_tokens)
                eval_time = time.time() - eval_start_time
                eval_msg = f"For validation data: loss = {eval_loss_epoch:.4f}, accuracy = {eval_accuracy_epoch:.8f}, Time: {eval_time:.2f}s"
                self.logger.info(eval_msg)
            
                # Check if this is the best model
                is_best = eval_loss_epoch < self.best_loss
                if is_best:
                    self.best_loss = eval_loss_epoch
                    self.logger.info(f"New best model found at epoch {epoch+1}")

                self.eval_loss.append(eval_loss_epoch)
                self.eval_accuracy.append(eval_accuracy_epoch)
                
                # Save checkpoint
                self._save_checkpoint(epoch, is_best=is_best)

            self.model.train()
        
        total_training_time = time.time() - training_start_time
        self.logger.info("============ Training Summary ============")
        final_msg = f"Training completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)"
        self.logger.info(final_msg)
        self.logger.info(f"Best validation loss: {self.best_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {max(self.eval_accuracy):.8f}")
        
        # Save final checkpoint
        self._save_checkpoint(self.num_epochs - 1, is_best=False)
        self.logger.info(f"Final training state saved")
        
        return {
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'eval_accuracy': self.eval_accuracy,
            'training_time': total_training_time,
            'best_loss': self.best_loss,
            'session_path': self.logger.cache_dir
        }


# Legacy script mode support - if trainer.py is run directly
if __name__ == "__main__":
    from ..utils.logger import Logger
    
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default_config.yaml')
    config = load_config(config_path)
    
    # Create logger for data generation
    cache_dir = config['training']['cache_dir']
    logger = Logger(cache_dir=cache_dir)
    
    # Extract data configuration
    n = config['data']['n']  # Dyck words semilength
    data_path = config['data'].get('data_path', None)  # None will use default path with n
    force_regenerate = config['data'].get('force_regenerate', False)
    
    # Generate Dyck words data with caching
    data = generate_dyck_data(n, data_path=data_path, force_regenerate=force_regenerate, logger=logger)
    
    # Create dataset object
    dataset = Dataset(data, batch_size=config['training']['batch_size'])
    
    # Update max_len based on actual data and architecture
    target_len = dataset.train_dataloader.dataset[1][1].shape[0]    # assuming they are all the same len
    input_len = dataset.train_dataloader.dataset[1][0].shape[0]
    if config['model']['architecture'] == 'decoder_only':
        # For decoder-only, we concatenate input + target, so need longer max_len
        max_len = input_len + target_len
    else:
        # For other architectures, use target length or max of input/target
        max_len = max(input_len, target_len)
    
    # Initialize model
    model = Transformer(
        src_vocab_size=config['model']['src_vocab_size'], 
        tgt_vocab_size=config['model']['tgt_vocab_size'], 
        d_model=config['model']['d_model'], 
        num_heads=config['model']['num_heads'], 
        d_ff=config['model']['d_ff'], 
        num_encoder_layers=config['model']['num_encoder_layers'], 
        num_decoder_layers=config['model']['num_decoder_layers'], 
        max_len=max_len, 
        dropout=config['model']['dropout'], 
        architecture=config['model']['architecture'], 
        is_sinusoidal=config['model']['is_sinusoidal']
    )
    
    # Initialize and run trainer
    trainer = Trainer(model, dataset, config)
    results = trainer.train()
    logger.info(f"Training completed. Results: {results}")