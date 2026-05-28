"""
app/model.py

PyTorch models for human action recognition from sequence keypoints.
Implements an LSTM with self-attention (ActionLSTM) and a Transformer (ActionTransformer).
"""

import math
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn


class ActionLSTM(nn.Module):
    """
    LSTM-based neural network with self-attention for sequence classification.
    """
    
    def __init__(
        self,
        input_size: int = 132,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initializes the ActionLSTM architecture.
        
        Args:
            input_size (int): Number of features per frame.
            hidden_size (int): Size of the LSTM hidden state.
            num_layers (int): Number of stacked LSTM layers.
            num_classes (int): Number of output action classes.
            dropout (float): Dropout probability.
            bidirectional (bool): Whether the LSTM is bidirectional.
        """
        super().__init__()
        
        self.num_directions = 2 if bidirectional else 1
        lstm_out_dim = hidden_size * self.num_directions
        
        # Input projection to map raw keypoints into a richer feature space
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Self-attention scoring layer
        self.attention_weights = nn.Linear(lstm_out_dim, 1)
        
        # Layer normalization applied to the attention output
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Applies scaled dot-product self-attention over the time dimension of LSTM outputs.
        
        Args:
            lstm_out (torch.Tensor): Output from the LSTM of shape (batch, seq_len, hidden_size * num_directions).
            
        Returns:
            torch.Tensor: Weighted sum of hidden states of shape (batch, hidden_size * num_directions).
        """
        # Calculate attention scores: (batch, seq_len, 1)
        scores = self.attention_weights(lstm_out)
        
        # Apply softmax over the time dimension (dim=1)
        weights = torch.softmax(scores, dim=1)
        
        # Compute weighted sum over time dimension: (batch, hidden_size * num_directions)
        context = torch.sum(weights * lstm_out, dim=1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ActionLSTM.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch, seq_len, input_size).
            
        Returns:
            torch.Tensor: Raw logits of shape (batch, num_classes).
        """
        # x shape: (batch_size, seq_len, input_size)
        
        # 1. Input projection
        x = self.input_proj(x)
        # x shape: (batch_size, seq_len, hidden_size)
        
        # 2. LSTM
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        
        # 3. Attention mechanism
        context = self._attention(lstm_out)
        # context shape: (batch_size, hidden_size * num_directions)
        
        # 4. Layer Normalization
        norm_context = self.layer_norm(context)
        # norm_context shape: (batch_size, hidden_size * num_directions)
        
        # 5. Classifier head
        logits = self.classifier(norm_context)
        # logits shape: (batch_size, num_classes)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer architecture.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.
        
        Args:
            d_model (int): The embedding dimension size.
            max_len (int): Maximum expected sequence length.
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).
            
        Returns:
            torch.Tensor: Position-encoded tensor.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class ActionTransformer(nn.Module):
    """
    Lightweight Transformer alternative for sequence classification.
    """
    
    def __init__(
        self,
        input_size: int = 132,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        max_seq_len: int = 30
    ):
        """
        Initializes the ActionTransformer architecture.
        
        Args:
            input_size (int): Number of features per frame.
            d_model (int): Transformer embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            num_classes (int): Number of output action classes.
            dropout (float): Dropout probability.
            max_seq_len (int): Maximum sequence length.
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ActionTransformer.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch, seq_len, input_size).
            
        Returns:
            torch.Tensor: Raw logits of shape (batch, num_classes).
        """
        # x shape: (batch_size, seq_len, input_size)
        
        # 1. Input projection
        x = self.input_proj(x)
        # x shape: (batch_size, seq_len, d_model)
        
        # 2. Positional encoding
        x = self.pos_encoder(x)
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x)
        # x shape: (batch_size, seq_len, d_model)
        
        # 4. Global average pooling over the sequence dimension (dim=1)
        x_pooled = torch.mean(x, dim=1)
        # x_pooled shape: (batch_size, d_model)
        
        # 5. Classifier head
        logits = self.classifier(x_pooled)
        # logits shape: (batch_size, num_classes)
        
        return logits


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts the total and trainable parameters of a PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model.
        
    Returns:
        Tuple[int, int]: Total parameters and trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model(
    model_type: str,
    num_classes: int,
    device: torch.device,
    **kwargs: Any
) -> nn.Module:
    """
    Instantiates and returns the chosen model, moved to the specified device.
    
    Args:
        model_type (str): Type of model to instantiate ('lstm' or 'transformer').
        num_classes (int): Number of output classes.
        device (torch.device): Target device (CPU or GPU).
        **kwargs: Additional keyword arguments for the model constructor.
        
    Returns:
        nn.Module: The instantiated model.
        
    Raises:
        ValueError: If an unsupported model_type is provided.
    """
    if model_type.lower() == 'lstm':
        model = ActionLSTM(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'transformer':
        model = ActionTransformer(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Expected 'lstm' or 'transformer'.")
        
    model = model.to(device)
    
    total, trainable = count_parameters(model)
    print(f"--- Model Summary: {model_type.upper()} ---")
    print(f"Total Parameters:     {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    
    return model


if __name__ == "__main__":
    # Smoke test for both models
    print("Running Smoke Test for Action Models...\n")
    
    batch_size = 32
    seq_len = 30
    input_size = 132
    num_classes = 5
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    print(f"Dummy Input Shape: {dummy_input.shape}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test LSTM
    lstm_model = get_model('lstm', num_classes=num_classes, device=device)
    dummy_input_device = dummy_input.to(device)
    lstm_output = lstm_model(dummy_input_device)
    print(f"LSTM Output Shape: {lstm_output.shape}\n")
    
    if lstm_output.shape != (batch_size, num_classes):
        raise AssertionError(f"LSTM Output shape is incorrect! Expected {(batch_size, num_classes)}, got {lstm_output.shape}")
        
    # Test Transformer
    transformer_model = get_model('transformer', num_classes=num_classes, device=device)
    transformer_output = transformer_model(dummy_input_device)
    print(f"Transformer Output Shape: {transformer_output.shape}\n")
    
    if transformer_output.shape != (batch_size, num_classes):
        raise AssertionError(f"Transformer Output shape is incorrect! Expected {(batch_size, num_classes)}, got {transformer_output.shape}")
        
    print("All assertions passed! Output shapes are fully correct.")
