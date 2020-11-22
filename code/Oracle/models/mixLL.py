from types import SimpleNamespace
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.model_loading import load_model

use_cuda = torch.cuda.is_available()

class mixLL(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.mlp = nn.Sequential(
                        nn.Linear(input_size+hidden_size, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 128),
                        nn.ReLU(),
                        nn.Linear(128, 3),
                        nn.Softmax()
                    )


    def forward(self, encodings, comodin):
        output, (hn, cn) = self.lstm(encodings)
        if output.size()[1] > 1:
            #print(output[-1][-2], output[-1][-2].size())
            #print(encodings.size())
            #exit(1)
            a = torch.cat([output[-1][-2], encodings[-1][-1]])
        else:
            a = torch.cat([comodin, encodings[-1][-1]]) 
        out = self.mlp(a)
        return out