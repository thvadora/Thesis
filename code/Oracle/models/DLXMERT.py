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

    """def forward(self, encodings, sizes):
        print("Esto toma como entrada: \n")
        print(encodings)
        lengths, ind = torch.sort(sizes, descending=True)
        #print(lengths)
        input_seq = encodings[ind]
        print("Reordena: (solo importa con batchsize > 1)\n")
        print(input_seq)
        packed = pack_padded_sequence(input_seq, list(lengths), batch_first=True)
        print("Secuencia empaquetada: \n")
        print(packed)
        output, states = self.lstm(packed)
        print("Salida de la LSTM, los hiden states de cada input: \n")
        print(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        print("Los desempaco, esto es la entrada a un preceptron que mapea cada hidden state a un vector de 3 (No, Si, n/a) \n")
        print(output)
        to = self.hidden2ans(output)
        print("Salida de perceptron: \n")
        print(to)
        out = self.sftmx(to)
        print("Le aplico Softmax para clasificacion: \n")
        print(out)
        out = torch.transpose(out, 2, 1)
        print("Traspongo para que este en el orden que asume la funcion de crossentropy de pytorch")
        print(out)
        return out"""
    
    def forward(self, encodings, sizes, debug=False):
        if debug:
            print("Entrada: ")
            print(encodings)
            print("Se lo damos de comer a la LSTM")       
        output, states = self.lstm(encodings)
        if debug:
            print("Output LSTM, para cada encoding tenemos su hidden state: ")
            print(output)
            print("Esta salida se lo paso a una capa Lineal que mapea cada hidden state a un vetor de tamaño 3")
        to = self.hidden2ans(output)
        if debug:
            print("Salida Lineal: ")
            print(to) 
            print("Luego aplico Softmax para clasificacion")
        out = self.sftmx(to)
        if debug:
            print("Resultado: ")
            print(out)
            print("Luego transpongo para que sea corde a lo que asume la loss function de pytorch")
        out = torch.transpose(out, 2, 1)
        if debug:
            print(out) 
        return out