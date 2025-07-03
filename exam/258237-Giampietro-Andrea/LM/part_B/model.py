import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class LM_LSTM_weight_tying(nn.Module):
    """
    Mode 1: LSTM + Weight Tying
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_weight_tying, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        # LSTM layer (no dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: share weights between embedding and output layer
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        # Embedding (no dropout)
        emb = self.embedding(input_sequence)
        
        # LSTM
        rnn_out, _ = self.lstm(emb)
        
        # Linear projection
        output = self.output(rnn_out).permute(0, 2, 1)
        
        return output


class LM_LSTM_variational_dropout(nn.Module):
    """
    Mode 2: LSTM + Weight Tying + Variational Dropout
    Mode 3: LSTM + Weight Tying + Variational Dropout + NT-AvSGD (same model, NT-AvSGD is handled externally)
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.3,
                 emb_dropout=0.3, n_layers=1):
        super(LM_LSTM_variational_dropout, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: share weights between embedding and output layer
        self.output.weight = self.embedding.weight

        # Dropout rates for variational dropout
        self.emb_dropout_rate = emb_dropout
        self.out_dropout_rate = out_dropout

    def forward(self, input_sequence):
        batch_size, seq_len = input_sequence.size()
        
        # Get embeddings
        emb = self.embedding(input_sequence)  # Shape: [batch_size, seq_len, emb_size]
        
        # Apply variational dropout to embeddings (same mask for all timesteps)
        if self.training and self.emb_dropout_rate > 0:
            emb = self.apply_variational_dropout(emb, self.emb_dropout_rate)
            
        # LSTM forward pass
        lstm_out, _ = self.lstm(emb)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Apply variational dropout to LSTM outputs (before final layer)
        if self.training and self.out_dropout_rate > 0:
            lstm_out = self.apply_variational_dropout(lstm_out, self.out_dropout_rate)
            
        # Project to output space and reshape for cross-entropy loss
        output = self.output(lstm_out).permute(0, 2, 1)  # Shape: [batch_size, output_size, seq_len]
        
        return output
    
    def apply_variational_dropout(self, x, p=0.3):
        """
        Apply variational dropout: same dropout mask across time dimension
        """
        if not self.training or p == 0.0:
            return x
            
        # Create mask: [batch, 1, feature_dim] -> same mask across all timesteps
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - p)
        mask = mask.div_(1 - p)  # Re-scale to maintain expected value
        
        return x * mask  # Broadcasting: same mask applied to all timesteps


class LM_LSTM_nt_avsgd(nn.Module):
    """
    Mode 3: LSTM + Weight Tying + Variational Dropout + NT-AvSGD
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.3,
                 emb_dropout=0.3, n_layers=1):
        super(LM_LSTM_nt_avsgd, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

        # Weight tying: share weights between embedding and output layer
        self.output.weight = self.embedding.weight

        # Dropout rates for variational dropout
        self.emb_dropout_rate = emb_dropout
        self.out_dropout_rate = out_dropout

    def forward(self, input_sequence):
        batch_size, seq_len = input_sequence.size()
        
        # Get embeddings
        emb = self.embedding(input_sequence)  # Shape: [batch_size, seq_len, emb_size]
        
        # Apply variational dropout to embeddings (same mask for all timesteps)
        if self.training and self.emb_dropout_rate > 0:
            emb = self.apply_variational_dropout(emb, self.emb_dropout_rate)
            
        # LSTM forward pass
        lstm_out, _ = self.lstm(emb)  # Shape: [batch_size, seq_len, hidden_size]
        
        # Apply variational dropout to LSTM outputs (before final layer)
        if self.training and self.out_dropout_rate > 0:
            lstm_out = self.apply_variational_dropout(lstm_out, self.out_dropout_rate)
            
        # Project to output space and reshape for cross-entropy loss
        output = self.output(lstm_out).permute(0, 2, 1)  # Shape: [batch_size, output_size, seq_len]
        
        return output
    
    def apply_variational_dropout(self, x, p=0.3):
        """
        Apply variational dropout: same dropout mask across time dimension
        """
        if not self.training or p == 0.0:
            return x
            
        # Create mask: [batch, 1, feature_dim] -> same mask across all timesteps
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - p)
        mask = mask.div_(1 - p)  # Re-scale to maintain expected value
        
        return x * mask  # Broadcasting: same mask applied to all timesteps