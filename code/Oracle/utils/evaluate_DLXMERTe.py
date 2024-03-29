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
from models.DLXMERT import DLXMERTe
from utils.config import load_config
from utils.model_loading import load_model
from utils.datasets.Oracle.DLXMERTOracleDataset import DLXMERTOracleDataset
from utils.vocab import create_vocab
from utils.evaluate_byclass import compute_bycategory

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
    #print('predictions: ', predictions)
    #print(predictions.size())
    #print('target: ', targets[0])
    #print(targets.size())
    predicted_classes = predictions.topk(1,dim=1)[1]
    #print('classes: ', predicted_classes.squeeze(1)[0])
    #print('nom: ', torch.eq(predicted_classes.squeeze(1)[0], targets[0]).sum().item())
    #print('denom: ', predicted_classes.squeeze(1)[0].size()[0])
    accuracy = (torch.eq(predicted_classes.squeeze(1)[0], targets[0]).sum().item())/predicted_classes.squeeze(1)[0].size()[0]
    #print(accuracy)
    #print(torch.eq(predicted_classes.squeeze(1), targets))
    #print(torch.eq(predicted_classes.squeeze(1), targets).size())
    return accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-bin_name", type=str, default="dlxmerte", help='Data Directory')
    parser.add_argument("-set", type=str, default="val")
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument("-add_bycat", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default='')
    args = parser.parse_args()

    onlyhist = False
    if args.dataset == 'historical':
        onlyhist = True
    
    #hiperparametros lr, dropout, batch_size, epochs
    epochs = 25
    lr = 0.00001
    dropout = 0
    batch_size = 1
    lstm_hidden = 100
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
        only_encodings = True,
        onlyhist=onlyhist
    )

    torch.set_grad_enabled(False)
    model.eval()

    dataloader = DataLoader(
                dataset=dataset,
                batch_size=1,
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
    fname = args.dataset+"dlxmert"+args.set+"predictions.csv"
    with open(fname, mode="w", encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Game ID", "Position", "qid", "Input", "GT Answer", "Model Answer"])
        stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
        for i_batch, batch in stream:
            encodings = batch['lxmertout']
            lengths = batch['sizes']
            mxl = max(lengths)
            answers = torch.narrow(batch['answers'], 1, 0, mxl)
            #print(answers[0][0].item())
            output = model(encodings, lengths)
            predicted_classes = output.topk(1,dim=1)[1]
            res = predicted_classes.squeeze(1)[0]
            gameid = batch['game_ids'].item()
            qids = batch['qids']
            rawin = batch['raw_q']
            #print(res)
            #break
            for index, x in enumerate(encodings[0]):
                newans = tok2ans[res[index].item()]
                realans = tok2ans[answers[0][index].item()]
                rawq = rawin[index][0]
                qid = qids[index].item()
                writer.writerow(
                    [
                        gameid,
                        index,
                        qid,
                        rawq,
                        realans,
                        newans
                    ]
                )
    
    if args.add_bycat:
        #compute_bycategory(fname)
        pass
    #df.to_csv('dlxmert'+args.set+'predictions.csv')
    #report accuray