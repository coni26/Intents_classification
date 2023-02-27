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
        self.name = 'RNN_' + str(hidden_dim) + '_' + str(n_layers)

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
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden
    
    
class RNNModel_bi(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel_bi, self).__init__()
        self.name = 'biRNN_' + str(hidden_dim) + '_' + str(n_layers)

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
        self.name = 'MLP_'
         
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, end_dim=1)
        #out = torch.sigmoid(self.fc1(x))
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0.2):
        super(GRUModel, self).__init__()
        self.name = 'GRU_' + str(hidden_dim) + '_' + str(n_layers)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, bias=True, dropout=dropout)   
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.gru(x, hidden)
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        #out = torch.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden
    

class GRUModel_bi(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0.0):
        super(GRUModel_bi, self).__init__()
        self.name = 'biGRU_' + str(hidden_dim) + '_' + str(n_layers)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, bias=True, bidirectional=True, dropout=dropout)   
        self.fc = nn.Linear(2 * hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.gru(x, hidden)

        out = out.contiguous().view(-1, int(2 * self.hidden_dim))
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        #out = torch.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden

    
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.name = 'LSTM_' + str(hidden_dim) + '_' + str(n_layers)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, bias=True)   
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.lstm(x, hidden)
        
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        #out = torch.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden
    
class LSTMModel_bi(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTMModel_bi, self).__init__()
        self.name = 'biLSTM_' + str(hidden_dim) + '_' + str(n_layers)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, bias=True, bidirectional=True)   
        self.fc = nn.Linear(2 * hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.lstm(x, hidden)

        out = out.contiguous().view(-1, int(2 * self.hidden_dim))
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        #out = torch.sigmoid(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(2 * self.n_layers, batch_size, self.hidden_dim).to(device))
        return hidden