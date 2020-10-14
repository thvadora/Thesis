import argparse
import datetime
import json
import sys
from collections import OrderedDict
from time import time
import gzip
import numpy as np
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
    parser.add_argument("-save_in", type=str, default="bin/Oracle", help='Data Directory')
    parser.add_argument("-bin_name", type=str, default="dlxmerte", help='Data Directory')
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

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    if use_cuda:
        model.cuda()
        model = DataParallel(model)

    dataset_train = DLXMERTOracleDataset(
        turns = 16,
        data_path = args.data_dir,
        sett = 'train',
        only_encodings = True
    )

    dataset_validation = DLXMERTOracleDataset(
        turns = 16,
        data_path = args.data_dir,
        sett = 'val',
        only_encodings = True
    )

    print("Initializing the optimizer...")
    num_batches_per_epoch = len(dataset_train) // batch_size
    num_total_batches = num_batches_per_epoch * epochs
    print("Number of batches per epoch: {}".format(num_batches_per_epoch))
    print("Total number of batches: {}".format(num_total_batches))

    for epoch in range(epochs):
        # Init logging variables
        start = time()
        loss, train_accuracy, val_accuracy = 0, 0, 0

        if use_cuda:
            train_loss = torch.cuda.FloatTensor()
            val_loss = torch.cuda.FloatTensor()
        else:
            train_loss = torch.FloatTensor()
            val_loss = torch.FloatTensor()

        for split, dataset in zip(['train', 'val'], [dataset_train, dataset_validation]):
            accuracy = []
            #print(split)
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0 if sys.gettrace() else 4,
                pin_memory=use_cuda
            )
            if split == 'train':
                torch.set_grad_enabled(True)
                model.train()
            else:
                torch.set_grad_enabled(False)
                model.eval()

            stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
            for i_batch, batch in stream:
                encodings = batch[0]
                answers = torch.reshape(batch[1][0], (batch_size, 16))
                output = model(Variable(encodings))
                loss = loss_function(output, Variable(answers).cuda() if use_cuda else Variable(answers)).unsqueeze(0)

                accuracy.append(calculate_accuracy_oracle(output, answers.cuda() if use_cuda else answers))
                
                stream.set_description("Train accuracy: {}".format(np.round(np.mean(accuracy), 2)))
                stream.refresh()  # to show immediately the update
                
                if split == 'train':
                    # Backprop and parameter update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss = torch.cat([train_loss, loss.data])
                else:
                    val_loss = torch.cat([val_loss, loss.data])

            if split == 'train':
                train_accuracy = np.mean(accuracy)
            elif split == 'val':
                val_accuracy = np.mean(accuracy)

        torch.save(model.state_dict(), os.path.join(args.save_in, ''.join(['oracle', args.bin_name, str(epoch)])))
        print("saving ", os.path.join(args.save_in, ''.join(['oracle', args.bin_name, str(epoch)])))

        print("%s, Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f, Training Accuracy %.5f, Validation Accuracy %.5f"
        %(args.bin_name, epoch, time()-start, torch.mean(train_loss), torch.mean(val_loss), train_accuracy, val_accuracy))
