#-------------------------------------------------------------------
# Default configuration for the Transformer model, training, and optimizer
#-------------------------------------------------------------------

# Model parameters
src_vocab_size = 4
tgt_vocab_size = 4
d_model = 128
num_heads = 4
d_ff = 256
num_encoder_layers = 3    # 3 works
num_decoder_layers = 3    # 3 works
max_len = 128
dropout = 0.1
architecture = 'encoder_decoder'
is_sinusoidal = False

# Training and optimizer parameters
batch_size = 32           
num_epochs = 100                
lr = 0.0001                     # Learning rate for optimizer
weight_decay = 0                # Weight decay for optimizer
