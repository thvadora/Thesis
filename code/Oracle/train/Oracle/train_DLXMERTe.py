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
                answers = torch.reshape(batch[1], (batch_size, 16))
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




"""
                # Calculate Loss
                loss = loss_function(pred_answer, Variable(answers).cuda() if exp_config['use_cuda'] else Variable(answers)).unsqueeze(0)

                # Calculate Accuracy
                accuracy.append(calculate_accuracy_oracle(pred_answer, answers.cuda() if exp_config['use_cuda'] else answers))

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

                # bookkeeping
                if split == 'train' and exp_config['logging']:
                    writer.add_scalar("Training/Batch Accuracy", accuracy[-1], train_batch_out)
                    writer.add_scalar("Training/Batch Loss", loss.data[0], train_batch_out)

                    train_batch_out += 1

                    if i_batch == 0:
                        for name, param in model.named_parameters():
                            writer.add_histogram("OracleParams/Oracle_" + name, param.data, epoch, bins='auto')

                        if epoch > 0 and epoch%5 == 0:
                            labels = list(OrderedDict(sorted({int(k):v for k,v in i2word.items()}.items())).values())
                            writer.add_embedding(model.module.word_embeddings.weight.data, metadata=labels, tag='oracle word embedding', global_step=int(epoch/5))

                        if epoch == 0:
                            writer.add_graph(model, pred_answer)


                elif split == 'val' and exp_config['logging']:
                    writer.add_scalar("Validation/Batch Accurarcy", accuracy[-1], valid_batch_out)
                    writer.add_scalar("Validation/Batch Loss", loss.data[0], valid_batch_out)
                    valid_batch_out += 1

            # bookkeeping
            if split == 'train':
                train_accuracy = np.mean(accuracy)
            elif split == 'val':
                val_accuracy = np.mean(accuracy)


        if exp_config['save_models']:
            if not os.path.exists(exp_config['save_models_path']):
                os.makedirs(exp_config['save_models_path'])
            torch.save(model.state_dict(), os.path.join(exp_config['save_models_path'], ''.join(['oracle', args.bin_name, exp_config['ts'], str(epoch)])))
            print("saving ", os.path.join(exp_config['save_models_path'], ''.join(['oracle', args.bin_name, exp_config['ts'], str(epoch)])))


        print("%s, Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f, Training Accuracy %.5f, Validation Accuracy %.5f"
        %(args.exp_name, epoch, time()-start, torch.mean(train_loss), torch.mean(val_loss), train_accuracy, val_accuracy))

        if exp_config['logging']:
            writer.add_scalar("Training/Epoch Loss", torch.mean(train_loss), epoch)
            writer.add_scalar("Training/Epoch Accuracy", train_accuracy, epoch)

            writer.add_scalar("Validation/Epoch Loss", torch.mean(val_loss), epoch)
            writer.add_scalar("Validation/Epoch Accuracy", val_accuracy, epoch)
"""