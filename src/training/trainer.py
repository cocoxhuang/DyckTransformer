from copy import deepcopy
import torch
import torch.nn as nn
import os

# Import configuration
from ..utils.config import load_config
# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default_config.yaml')
config = load_config(config_path)

# Extract configuration values
src_vocab_size = config['model']['src_vocab_size']
tgt_vocab_size = config['model']['tgt_vocab_size']
d_model = config['model']['d_model']
num_heads = config['model']['num_heads']
d_ff = config['model']['d_ff']
num_encoder_layers = config['model']['num_encoder_layers']
num_decoder_layers = config['model']['num_decoder_layers']
max_len = config['model']['max_len']
dropout = config['model']['dropout']
architecture = config['model']['architecture']
is_sinusoidal = config['model']['is_sinusoidal']

batch_size = config['training']['batch_size']
num_epochs = config['training']['num_epochs']
lr = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']

# Import custom classes
from ..data.dataset import Dataset
from ..model.transformer import Transformer
from .evaluator import Evaluator

# TODO: Define or import the 'data' variable
# data = your_training_data_here
data = None  # Replace with actual training data

# create a dataset object
dataset = Dataset(data, batch_size=batch_size)

max_len = dataset.train_dataloader.dataset[1][1].shape[0] 
# max_new_tokens = 1
# Initialize model, optimizer, and loss function
# Calculate vocab size after shifting and adding BOS/EOS
model = Transformer(
    src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=d_model, num_heads=num_heads, 
    d_ff=d_ff, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, max_len=max_len, 
    dropout=dropout, architecture=architecture, is_sinusoidal=is_sinusoidal
)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) 
criterion = nn.CrossEntropyLoss()                                           # for training loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training using device: {device}.")
model.to(device)
model.train()
# print many parameters of the model
print("============ Model Information ============")
print(f"Model is {'decoder only' if model.architecture == 'decoder_only' else 'encoder-decoder' if model.architecture == 'encoder_decoder' else 'encoder only'}.")
print(f"Training using device: {device}.")
print(f"Model has {num_encoder_layers} encoder layers, {num_decoder_layers} decoder heads, embedding dimension is {d_model}.")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Model is using {'sinusoidal' if model.is_sinusoidal else 'learnable'} positional encoding.")

# training loop for the Transformer model
train_loss = []
eval_loss = []
train_accuracy = []
eval_accuracy = []
max_new_tokens = dataset.train_dataloader.dataset[1][1].shape[0] - 1  # Exclude BOS token, so -1

# Create evaluator instance
evaluator = Evaluator(model, dataset.eval_dataloader, criterion, device)

print("============ Start Training ============")
# Training loop with teacher forcing and evaluation
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in dataset.train_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Teacher forcing: use shifted targets as decoder input
        optimizer.zero_grad()
        outputs = model(
            src=inputs,
            tgt=targets[:, :-1]  # exclude last token
        )
        
        # Calculate loss on shifted output
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            targets[:, 1:].contiguous().view(-1)  # exclude first token
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Average loss over batches (not dataset length)
    avg_train_loss = total_loss / len(dataset.train_dataloader)
    train_loss.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    with torch.no_grad():
        model.eval()
        
        # train_loss_epoch, train_accuracy_epoch = evaluate(model, train_dataloader, device=device, criterion=criterion)
        eval_loss_epoch, eval_accuracy_epoch = evaluator.evaluate(max_new_tokens)
        # print(f"For training data: loss = {train_loss_epoch:.4f}, accuracy = {train_accuracy_epoch:.4f}")
        print(f"For validation data: loss = {eval_loss_epoch:.4f}, accuracy = {eval_accuracy_epoch:.8f}")

        train_loss.append(avg_train_loss)
        eval_loss.append(eval_loss_epoch)
        # train_accuracy.append(train_accuracy_epoch)
        eval_accuracy.append(eval_accuracy_epoch)
    
        # save the best model
        if epoch == 0 or eval_loss_epoch < min(eval_loss):
            best_model = deepcopy(model.state_dict())
            print("Best model saved.")
            # Save the best model
            torch.save(best_model, 'best_transformer_model.pth')

    model.train()