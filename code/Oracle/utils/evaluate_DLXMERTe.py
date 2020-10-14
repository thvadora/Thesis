import argparse
import datetime
import json
import sys
from collections import OrderedDict
from time import time
import gzip
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from lxmert.src.lxrt.optimization import BertAdam
from models.DLXMERT import DLXMERTe
from utils.config import load_config
from utils.model_loading import load_model
from utils.datasets.Oracle.DLXMERTOracleDataset import DLXMERTOracleDataset
from utils.vocab import create_vocab

use_cuda = torch.cuda.is_available()

def calculate_accuracy_oracle(predictions, targets):
    """
    :param prediction: NxCxdialogsize
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data
    #print(predictions)
    #print(targets)
    predicted_classes = predictions.topk(1,dim=1)[1]
    accuracy = (torch.eq(predicted_classes.squeeze(1), targets).sum().item())/(targets.size(0)*targets.size(1))
    #print(torch.eq(predicted_classes.squeeze(1), targets))
    #print(torch.eq(predicted_classes.squeeze(1), targets).size())
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-bin_name", type=str, default="dlxmerte", help='Data Directory')
    parser.add_argument("-lxmertcsv", type=str, default="lxmertvalpredictions.csv")
    parser.add_argument("-set", type=str, default="val")
    parser.add_argument("-load_bin_path", type=str)
    args = parser.parse_args()
    
    #hiperparametros lr, dropout, batch_size, epochs
    epochs = 20
    lr = 0.00001
    dropout = 0
    batch_size = 32
    lstm_hidden = 6
    lstm_layers = 6

    with gzip.open('./data/guesswhat.train.jsonl.gz') as file:
        gw_train = file
    with gzip.open('./data/guesswhat.valid.jsonl.gz') as file:
        gw_valid = file
    with open(os.path.join(args.data_dir, 'qid2pos_train.json')) as file:
        qid2pos_train = json.load(file)
    with open(os.path.join(args.data_dir, 'qid2pos_val.json')) as file:
        qid2pos_valid = json.load(file)

    # Init Model, Loss Function and Optimizer
    model = DLXMERTe(
        input_size = 768,
        hidden_size = lstm_hidden, 
        num_layers = lstm_layers
    )

    if use_cuda:
        model.cuda()
        model = DataParallel(model)
    
    model.load_state_dict(torch.load(args.load_bin_path))

    dataset= DLXMERTOracleDataset(
        turns = 16,
        data_path = args.data_dir,
        sett = args.set,
        only_encodings = True
    )

    torch.set_grad_enabled(False)
    model.eval()

    dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                num_workers=0 if sys.gettrace() else 4,
                pin_memory=use_cuda
            )
    
    df = pd.read_csv(args.lxmertcsv)

    tok2ans = {
        0 : 'No',
        1 : 'Yes',
        2 : 'N/A'
    }

    stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    for i_batch, batch in stream:
        encoding = batch[0]
        dialogs = 0
        answers = torch.reshape(batch[1][0], (1, 16)).squeeze(0)
        gameid = batch[1][1][0]
        output = model(Variable(encoding))
        predicted_classes = output.topk(1,dim=1)[1].squeeze(0).squeeze(0)
        dialog_tope = 0
        for index, x in enumerate(encoding[0]):
            if x.sum() != 0:
                newans = tok2ans[predicted_classes[index].item()]
                df.loc[(df['Game ID']==gameid) & df['Position']==index, 'Model Answer'] = newans
                dialog_tope += 1
            else:
                break         
            if dialog_tope > 16:
                print(">16")
    df.to_csv('dlxmert'+args.set+'predictions.csv')