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
                            batch_first=True,
                            dropout=dropout)
        self.hidden2ans = nn.Linear(hidden_size, 3)
        self.sftmx = nn.Softmax(dim=2)

    def forward(self, batch):
        lengths, ind = torch.sort(batch[0][1], descending=True)
        input_seq = batch[0][0][ind]
        packed = pack_padded_sequence(input_seq, list(lengths), batch_first=True)
        #self.lstm.flatten_parameters()
        #print(type(encoded_dialog))
        output, states = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        to = self.hidden2ans(output)
        out = self.sftmx(to)
        out = torch.transpose(out, 2, 1)
        return out