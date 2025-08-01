import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, dropout, bidirectional, apply_dropout, n_layer=1, pad_index=0):        
        super(ModelIAS, self).__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder =  nn.LSTM(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        #Flags to control dropout and bidirectional
        self.apply_dropout = apply_dropout
        self.bidirectional = bidirectional
        
        if self.apply_dropout:
            self.apply_dropout = nn.Dropout(dropout)
            
        if bidirectional:
            hid_size=hid_size*2
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # Apply apply_dropout
        if self.apply_dropout:
            utt_encoded = self.apply_dropout(utt_encoded)

        # Get the last hidden state
        if self.utt_encoder.bidirectional:
            last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        else:
            last_hidden = last_hidden[-1,:,:]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent

