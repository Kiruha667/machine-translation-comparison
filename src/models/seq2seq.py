"""
Seq2Seq model with Attention mechanism
"""

import torch
import torch.nn as nn
import random
from typing import Tuple


class Encoder(nn.Module):
    """LSTM-based Encoder"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src: Source tensor (batch_size, seq_len)
            
        Returns:
            outputs: Encoder outputs (batch_size, seq_len, hidden_dim)
            hidden: Final hidden state (1, batch_size, hidden_dim)
            cell: Final cell state (1, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau Attention mechanism"""
    
    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Dimension of hidden state
        """
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: Decoder hidden state (batch_size, hidden_dim)
            encoder_outputs: Encoder outputs (batch_size, src_len, hidden_dim)
            
        Returns:
            attention_weights: Attention weights (batch_size, src_len)
        """
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state for each source position
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention weights
        attention = self.v(energy).squeeze(2)
        
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM-based Decoder with Attention"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            dropout: Dropout probability
        """
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input: torch.Tensor, hidden: torch.Tensor, 
                cell: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple:
        """
        Args:
            input: Input token (batch_size) or (batch_size, 1)
            hidden: Previous hidden state (1, batch_size, hidden_dim)
            cell: Previous cell state (1, batch_size, hidden_dim)
            encoder_outputs: Encoder outputs (batch_size, src_len, hidden_dim)
            
        Returns:
            prediction: Output predictions (batch_size, vocab_size)
            hidden: New hidden state
            cell: New cell state
            attn_weights: Attention weights (batch_size, src_len)
        """
        # Ensure input is 2D
        if input.dim() == 1:
            input = input.unsqueeze(1)
        
        # Embed input
        embedded = self.dropout(self.embedding(input))
        
        # Calculate attention weights
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        
        # Calculate context vector
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Make prediction
        prediction = self.fc(output.squeeze(1))
        
        return prediction, hidden, cell, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    """Complete Seq2Seq model with Attention"""
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        """
        Args:
            encoder: Encoder model
            decoder: Decoder model
            device: Device to run on (cuda/cpu)
        """
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            src: Source tensor (batch_size, src_len)
            tgt: Target tensor (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Model predictions (batch_size, tgt_len, vocab_size)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode source sentence
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to decoder is <sos> token
        input = tgt[:, 0]
        
        for t in range(1, tgt_len):
            # Decode
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            
            # Store output
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get highest predicted token
            top1 = output.argmax(1)
            
            # Use ground truth or prediction
            input = tgt[:, t] if teacher_force else top1
        
        return outputs
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_seq2seq_model(src_vocab_size: int, tgt_vocab_size: int,
                         embedding_dim: int = 256, hidden_dim: int = 512,
                         dropout: float = 0.1, device: torch.device = None) -> Seq2Seq:
    """
    Create and initialize Seq2Seq model
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        dropout: Dropout probability
        device: Device to run on
        
    Returns:
        Initialized Seq2Seq model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim, dropout)
    decoder = Decoder(tgt_vocab_size, embedding_dim, hidden_dim, dropout)
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    print(f"Seq2Seq model created:")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    
    return model


if __name__ == '__main__':
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_seq2seq_model(1000, 800, device=device)
    
    # Test forward pass
    src = torch.randint(0, 1000, (32, 10)).to(device)
    tgt = torch.randint(0, 800, (32, 12)).to(device)
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Should be (32, 12, 800)
