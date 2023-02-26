import os 
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='tanh', bias=True)   
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        #out = torch.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    
class RNNModel_bi(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel_bi, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='tanh', bias=True, bidirectional=True)   
        self.fc = nn.Linear(2 * hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, int(2 * self.hidden_dim))
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        #out = torch.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden
    
    
    
class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
         
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, end_dim=1)
        #out = torch.sigmoid(self.fc1(x))
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out
    
