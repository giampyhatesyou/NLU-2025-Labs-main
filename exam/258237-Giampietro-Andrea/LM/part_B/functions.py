# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np
# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def collate_fn(data, pad_token):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)


# NT-AvSGD implementation that stores state across epochs
class NTAvSGDTracker:
    def __init__(self, model, dev_loader, criterion_eval, non_monotone_interval=5):
        self.model = model
        self.dev_loader = dev_loader
        self.criterion_eval = criterion_eval
        self.non_monotone_interval = non_monotone_interval
        
        # State variables
        self.validation_history = []  # Store validation perplexities
        self.weight_history = []      # Store model weights for averaging
        self.triggered = False        # Whether averaging has been triggered
        self.T = 0                   # Trigger point (epoch to start averaging from)
        
    def validation_check(self, epoch):
        """Call this at the end of each epoch"""
        # Evaluate on dev set
        self.model.eval()
        ppl_dev, _ = eval_loop(self.dev_loader, self.criterion_eval, self.model)
        
        # Store validation result
        self.validation_history.append(ppl_dev)
        
        # Store current weights (deep copy)
        current_weights = {name: param.data.clone().cpu() for name, param in self.model.named_parameters()}
        self.weight_history.append(current_weights)
        
        # Check non-monotonic condition
        if not self.triggered and len(self.validation_history) > self.non_monotone_interval:
            # Check if current validation is worse than the best in recent history
            recent_best = min(self.validation_history[:-1])  # Exclude current
            current_val = self.validation_history[-1]
            
            if current_val > recent_best:
                self.triggered = True
                self.T = max(0, len(self.validation_history) - 2)  # Start averaging from previous epoch
                print(f"NT-AvSGD triggered at epoch {epoch}, starting averaging from epoch {self.T}")
        
        self.model.train()
        return ppl_dev
    
    def finalize(self):
        """Apply weight averaging at the end of training"""
        if not self.triggered or len(self.weight_history) <= self.T:
            print("NT-AvSGD was not triggered or insufficient history")
            return
        
        # Average weights from T to end
        weights_to_average = self.weight_history[self.T:]
        
        if len(weights_to_average) < 2:
            print("Not enough weights to average")
            return
        
        # Compute average weights
        averaged_weights = {}
        for name in weights_to_average[0].keys():
            weight_stack = torch.stack([w[name] for w in weights_to_average])
            averaged_weights[name] = torch.mean(weight_stack, dim=0)
        
        # Update model with averaged weights
        for name, param in self.model.named_parameters():
            if name in averaged_weights:
                param.data.copy_(averaged_weights[name].to(param.device))
        
        print(f"Applied NT-AvSGD averaging over {len(weights_to_average)} checkpoints")


def train_with_AvSGD(data, optimizer, criterion, model, clip=5):
    """
    Standard training loop - NT-AvSGD logic is handled externally
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array)/sum(number_of_tokens)



def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)