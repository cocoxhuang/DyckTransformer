from copy import deepcopy
import torch
from ..utils.config import load_config
import numpy as np
import random
import os

class Dataset:
    def __init__(self, data, seed = 42, batch_size=32):
        self.data = {
            'inputs': data['inputs'].clone(),
            'targets': data['targets'].clone()
        }
        self.bos_index = 0  # Beginning of sequence token index
        self.eos_index = 1  # End of sequence token index
        self.batch_size = batch_size
        self.train_dataloader, self.eval_dataloader = self.create_train_val_dataloader()
        # Move dictionary creation to AFTER data transformation
        self.dictionary = self._create_dictionary()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _create_dictionary(self):
        vocab_dict = {
            self.bos_index: 'bos',
            self.eos_index: 'eos'
        }
        
        # Now use the transformed data's max value
        max_token_id = max(self.data['inputs'].max().item(), self.data['targets'].max().item())
        
        # Map the actual token IDs that exist in the transformed data
        for token_id in range(2, max_token_id + 1):  # Start from 2 since 0,1 are BOS/EOS
            original_value = token_id - 2            # Reverse the +2 shift to get original value
            vocab_dict[token_id] = original_value
            
        return vocab_dict

    def create_dataloader(self, data):
        dataset = torch.utils.data.TensorDataset(data['inputs'], data['targets'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def create_train_val_dataloader(self):
        self.data['inputs'] = self.data['inputs'] + max(self.bos_index, self.eos_index) + 1
        self.data['targets'] = self.data['targets'] + max(self.bos_index, self.eos_index) + 1
        self.data['inputs'] = torch.cat((torch.full((self.data['inputs'].shape[0], 1), self.bos_index, dtype=torch.long), self.data['inputs']), dim=1)
        self.data['targets'] = torch.cat((torch.full((self.data['targets'].shape[0], 1), self.bos_index, dtype=torch.long), self.data['targets']), dim=1)
        
        shuffle_indices = torch.randperm(len(self.data['inputs']))
        train_idx = int(0.8 * len(self.data['inputs']))
        eval_idx = len(self.data['inputs']) - train_idx
        self.data['inputs'] = self.data['inputs'][shuffle_indices]
        self.data['targets'] = self.data['targets'][shuffle_indices]
        train_inputs = self.data['inputs'][:train_idx]
        train_targets = self.data['targets'][:train_idx]
        eval_inputs = self.data['inputs'][train_idx:]
        eval_targets = self.data['targets'][train_idx:]

        eval_size = 10000
        if eval_inputs.shape[0] > eval_size:
            eval_inputs = eval_inputs[:eval_size]
            eval_targets = eval_targets[:eval_size]

        train_dataloader = self.create_dataloader({'inputs': train_inputs, 'targets': train_targets})
        eval_dataloader = self.create_dataloader({'inputs': eval_inputs, 'targets': eval_targets})
        return train_dataloader, eval_dataloader