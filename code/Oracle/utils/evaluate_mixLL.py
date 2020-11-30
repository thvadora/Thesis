import argparse
import datetime
import json
import csv
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
from torch.nn.utils.rnn import pack_padded_sequence

from lxmert.src.lxrt.optimization import BertAdam
from models.mixLL import mixLL
from utils.config import load_config
from utils.model_loading import load_model
from utils.datasets.Oracle.POSDLXMERTOracleDataset import POSDLXMERTOracleDataset
from utils.vocab import create_vocab
from utils.evaluate_byclass import compute_bycategory

use_cuda = torch.cuda.is_available()

def calculate_accuracy_oracle(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    predicted_classes = predictions.topk(1)[1]
    accuracy = torch.eq(predicted_classes.squeeze(1), targets).sum().item()/targets.size(0)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-bin_name", type=str, default="dlxmerte", help='Data Directory')
    parser.add_argument("-set", type=str, default="val")
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument("-add_bycat", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("-onlyobj", type=bool, default=False, help='Data Directory')
    args = parser.parse_args()
    
    #hiperparametros lr, dropout, batch_size, epochs
    epochs = 50
    lr = 0.001
    dropout = 0
    batch_size = 1
    lstm_hidden = 150
    lstm_layers = 1

    with gzip.open('./data/guesswhat.train.jsonl.gz') as file:
        gw_train = file
    with gzip.open('./data/guesswhat.valid.jsonl.gz') as file:
        gw_valid = file
    with open(os.path.join(args.data_dir, 'qid2pos_train.json')) as file:
        qid2pos_train = json.load(file)
    with open(os.path.join(args.data_dir, 'qid2pos_val.json')) as file:
        qid2pos_valid = json.load(file)

    # Init Model, Loss Function and Optimizer
    model = mixLL(
        input_size = 768,
        hidden_size = lstm_hidden, 
        num_layers = lstm_layers
    )

    if use_cuda:
        model.cuda()
        model = DataParallel(model)
    
    model.load_state_dict(torch.load(args.load_bin_path))
    #model.load_state_dict(torch.load('./bin/Oracle/oracledlxmert8'))
    torch.set_grad_enabled(False)
    model.eval()

    dataset= POSDLXMERTOracleDataset(
        data_path = args.data_dir,
        sett = 'test',
        onlyobj = args.onlyobj
    )

    dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0 if sys.gettrace() else 4,
                pin_memory=use_cuda
            )

    tok2ans = {
        0 : 'No',
        1 : 'Yes',
        2 : 'N/A'
    }

    ob = ""
    if args.onlyobj:
        ob = "obj"
    fname = args.dataset+ob+"mixLL"+args.set+"predictions.csv"
    with open(fname, mode="w", encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Game ID", "Position", "qid", "Input", "GT Answer", "Model Answer"])
        stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
        for i_batch, batch in stream:
            encodings = batch['encodings']
            answers = batch['ans']
            comodin = torch.zeros(size=(lstm_hidden,))
            output = model(encodings, comodin)
            output = output.unsqueeze(0)
            predicted_classes = output.topk(1,dim=1)[1]
            res = predicted_classes.squeeze(1)[0].cpu().item()
            gameid = batch['game_id'].item()
            qid = batch['qid'].item()
            rawin = batch['question'][0]
            index = batch['pos'].item()
            gtans = answers.item()
            newans = tok2ans[res]
            realans = tok2ans[gtans]
            writer.writerow(
                [
                    gameid,
                    index,
                    qid,
                    rawin,
                    realans,
                    newans
                ]
            )