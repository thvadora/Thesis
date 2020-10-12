from types import SimpleNamespace
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget as LXMERT
from utils.model_loading import load_model

use_cuda = torch.cuda.is_available()

class DLXMERTe(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)
        self.hidden2ans = nn.Linear(hidden_size, 3)
        self.sftmx = nn.Softmax(dim=2)

    def forward(self, encoded_dialog):
        self.lstm.flatten_parameters()
        output, (h_n, c_n)  = self.lstm(encoded_dialog)
        to = self.hidden2ans(output)
        out = self.sftmx(to)
        out = torch.transpose(out, 2, 1)
        return out