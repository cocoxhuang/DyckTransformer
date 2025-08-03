import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, is_sinusoidal=False):
        super().__init__()
        self.is_sinusoidal = is_sinusoidal
        if is_sinusoidal:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Embedding(max_len, d_model)
            self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        if self.is_sinusoidal:
            return x + self.pe[:x.size(1)]
        else:
            position_ids = self.position_ids[:x.size(1)]
            return x + self.pe(position_ids).unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, srt_dim, d_model, num_heads, dropout, is_causal=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.src_dim = srt_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.last_attn_scores = None
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False

    def forward(self, q, k=None, v=None, mask=None):
        if k is None and v is None:
            k = v = q
        elif v is None:
            v = k
            
        batch_size = q.size(0)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        self.flash = False
        if self.flash:
            return ValueError("Flash attention is not working at the moment.")
        else:
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            if self.is_causal and mask is None:
                causal_mask = torch.ones_like(scores).triu(diagonal=1).bool()
                scores = scores.masked_fill(causal_mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            self.last_attn_scores = attn.detach()
            output = torch.matmul(self.attn_dropout(attn), v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, src_dim, d_model, num_heads, d_ff, dropout=0.1, is_decoder=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(src_dim, d_model, num_heads, dropout, is_causal=is_decoder)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.is_decoder = is_decoder
        if is_decoder:
            self.cross_attn = MultiHeadAttention(src_dim, d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
        
        self.ff = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, mask=tgt_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        if self.is_decoder and encoder_output is not None:
            attn_output = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
            x = x + self.dropout2(attn_output)
            x = self.norm2(x)
        
        ff_output = self.ff(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size=None, tgt_vocab_size=None, d_model=128, num_heads=4, 
                 d_ff=256, num_encoder_layers=3, num_decoder_layers=3, max_len=128, 
                 dropout=0.1, architecture='encoder_decoder', is_sinusoidal=False):
        super().__init__()
        self.architecture = architecture
        self.is_sinusoidal = is_sinusoidal
        
        if architecture != 'decoder_only':
            assert src_vocab_size is not None, "Source vocabulary size must be provided for encoder/encoder-decoder architectures"
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        if architecture != 'encoder_only':
            if architecture == 'encoder_decoder' and tgt_vocab_size is None:
                tgt_vocab_size = src_vocab_size
            assert tgt_vocab_size is not None, "Target vocabulary size must be provided for decoder/encoder-decoder architectures"
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        if architecture != 'decoder_only':
            self.src_pos_encoding = PositionalEncoding(d_model, max_len, is_sinusoidal)
        if architecture != 'encoder_only':
            self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, is_sinusoidal)
            
        if architecture != 'decoder_only':
            self.encoder = nn.ModuleList([
                TransformerBlock(d_model, d_model, num_heads, d_ff, dropout, is_decoder=False)
                for _ in range(num_encoder_layers)
            ])
        
        if architecture != 'encoder_only':
            self.decoder = nn.ModuleList([
                TransformerBlock(d_model, d_model, num_heads, d_ff, dropout, is_decoder=True)
                for _ in range(num_decoder_layers)
            ])
        
        if architecture != 'encoder_only':
            self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        else:
            self.fc_out = nn.Linear(d_model, src_vocab_size)
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src=None, tgt=None, src_mask=None, tgt_mask=None):
        if self.architecture == 'encoder_only':
            return self._forward_encoder(src, src_mask)
        elif self.architecture == 'decoder_only':
            return self._forward_decoder(tgt, tgt_mask)
        else:
            return self._forward_encoder_decoder(src, tgt, src_mask, tgt_mask)
    
    def _forward_encoder(self, src, src_mask=None):
        assert src is not None, "Source sequence must be provided for encoder-only architecture"
        
        src = self.src_embedding(src)
        src = self.src_pos_encoding(src)
        
        for layer in self.encoder:
            src = layer(src, src_mask=src_mask)
        
        return self.fc_out(src)
    
    def _forward_decoder(self, tgt, tgt_mask=None):
        assert tgt is not None, "Target sequence must be provided for decoder-only architecture"

        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos_encoding(tgt)
        
        for layer in self.decoder:
            tgt = layer(tgt, tgt_mask=tgt_mask)
        
        return self.fc_out(tgt)
    
    def _forward_encoder_decoder(self, src, tgt, src_mask=None, tgt_mask=None):
        assert src is not None, "Source sequence must be provided for encoder-decoder architecture"
        assert tgt is not None, "Target sequence must be provided for encoder-decoder architecture"

        src_emb = self.src_embedding(src)
        src_emb = self.src_pos_encoding(src_emb)
        for layer in self.encoder:
            src_emb = layer(src_emb, src_mask=src_mask)
        
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.tgt_pos_encoding(tgt_emb)
        
        for layer in self.decoder:
            tgt_emb = layer(tgt_emb, encoder_output=src_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        
        return self.fc_out(tgt_emb)
    
    def generate(self, src, tgt, max_new_tokens, temperature=1.0, top_k=None):
        if self.architecture == 'encoder_only':
            raise ValueError("Generation is not supported for encoder-only models")
        assert tgt is not None, "Target sequence must be provided for generation"
            
        for _ in range(max_new_tokens):
            if self.architecture == 'decoder_only':
                logits = self(tgt=tgt)
            else:
                logits = self(src=src, tgt=tgt)
            
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tgt = torch.cat((tgt, next_token), dim=1)
        
        return tgt