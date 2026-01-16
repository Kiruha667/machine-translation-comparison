"""
Transformer model for machine translation
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Complete Transformer model"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 4, num_decoder_layers: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_len: int = 5000):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate mask for target sequence (prevents looking ahead)
        
        Args:
            sz: Sequence length
            
        Returns:
            Mask tensor
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Source tensor (batch_size, src_len)
            tgt: Target tensor (batch_size, tgt_len)
            
        Returns:
            Output predictions (batch_size, tgt_len, vocab_size)
        """
        # Create masks
        src_key_padding_mask = (src == 0)  # Mask padding tokens
        tgt_key_padding_mask = (tgt == 0)
        
        tgt_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Embed and add positional encoding
        src_emb = self.dropout(
            self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        )
        tgt_emb = self.dropout(
            self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        )
        
        # Pass through transformer
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        return self.fc_out(output)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_model(src_vocab_size: int, tgt_vocab_size: int,
                             d_model: int = 512, nhead: int = 8,
                             num_encoder_layers: int = 4,
                             num_decoder_layers: int = 4,
                             dim_feedforward: int = 2048,
                             dropout: float = 0.1,
                             device: torch.device = None) -> TransformerModel:
    """
    Create and initialize Transformer model
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout probability
        device: Device to run on
        
    Returns:
        Initialized Transformer model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    print(f"Transformer model created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {nhead}")
    print(f"  Encoder layers: {num_encoder_layers}")
    print(f"  Decoder layers: {num_decoder_layers}")
    print(f"  Device: {device}")
    
    return model


if __name__ == '__main__':
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_transformer_model(1000, 800, device=device)
    
    # Test forward pass
    src = torch.randint(0, 1000, (32, 10)).to(device)
    tgt = torch.randint(0, 800, (32, 12)).to(device)
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Should be (32, 12, 800)
