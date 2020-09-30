from types import SimpleNamespace
import copy

import argparse
import csv
import datetime
import json
import sys
from time import time
import h5py

import numpy as np
import os
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget
from utils.config import load_config
from utils.datasets.Oracle.LXMERTOracleDataset import LXMERTOracleDataset
from utils.model_loading import load_model
from utils.vocab import create_vocab
from utils.evaluate_byclass import compute_bycategory

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget as LXMERT
from utils.model_loading import load_model

use_cuda = torch.cuda.is_available()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class GEN(nn.Module):

    def __init__(self, lxmert_oracle_dict, lxmert_oracle_weights, pretrain_size):
        super().__init__()
        self.lxmert = LXMERT(
            no_words            = lxmert_oracle_dict['no_words'],
            no_words_feat       = lxmert_oracle_dict['no_words_feat'],
            no_categories       = lxmert_oracle_dict['no_categories'],
            no_category_feat    = lxmert_oracle_dict['no_category_feat'],
            no_hidden_encoder   = lxmert_oracle_dict['no_hidden_encoder'],
            mlp_layer_sizes     = lxmert_oracle_dict['mlp_layer_sizes'],
            no_visual_feat      = lxmert_oracle_dict['no_visual_feat'],
            no_crop_feat        = lxmert_oracle_dict['no_crop_feat'],
            dropout             = lxmert_oracle_dict['dropout'],
            inputs_config       = lxmert_oracle_dict['inputs_config'],
            scale_visual_to     = lxmert_oracle_dict['scale_visual_to'],
            lxmert_encoder_args = lxmert_oracle_dict['lxmert_encoder_args']
        )
        self.lxmert = load_model(self.lxmert, lxmert_oracle_weights, use_dataparallel=use_cuda)
        
        #encoder only
        self.lxmert_encoder = copy.deepcopy(self.lxmert)
        self.lxmert_encoder.module.mlp = Identity()
        self.lxmert_encoder.eval()

        #extract crossAtt
        self.extractions = {}
        if pretrain_size == 'small':
            self.lxmert_encoder.module.lxrt_encoder.model.bert.encoder.x_layers[1].visual_attention.output.dense.register_forward_hook(self.extract('crossAtt'))
        if pretrain_size == 'big':
            self.lxmert_encoder.module.lxrt_encoder.model.bert.encoder.x_layers[4].visual_attention.output.dense.register_forward_hook(self.extract('crossAtt'))

    def extract(self, name):
        def hook(model, input, output):
            self.extractions[name] = output.detach()
        return hook

    def forward(self, questions, obj_categories, spatials, crop_features, visual_features, lengths,
                history_raw, fasterrcnn_features, fasterrcnn_boxes, target_bbox):
        lxmert_out = self.lxmert_encoder(questions,
                obj_categories,
                spatials,
                crop_features,
                visual_features,
                lengths,
                history_raw,
                fasterrcnn_features,
                fasterrcnn_boxes,
                target_bbox
            )
        crossAtt = self.extractions['crossAtt']
        return (lxmert_out, crossAtt)


def calculate_accuracy(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    accuracies = []
    predicted_classes = predictions.topk(1)[1]
    for accuracy in torch.eq(predicted_classes.squeeze(1), targets):
        accuracies.append(accuracy.item())
    return accuracies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="small", help='Config [small or big]')
    parser.add_argument("-img_feat", type=str, default="rss", help='Select "vgg" or "res" as image features')
    parser.add_argument("-set", type=str, default="test", help='Select train, val o test')
    parser.add_argument("-add_bycat", type=bool, default=True)
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("--preloaded", type=bool, default=False)
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument("-save_in", type=str)

    args = parser.parse_args()
    config_file = 'config/Oracle/config_small.json'
    
    if args.config == "big":
        config_file = 'config/Oracle/config.json'

    config = load_config(config_file)

    # Experiment Settings

    exp_config = config['exp_config']
    exp_config['img_feat'] = args.img_feat.lower()
    exp_config['use_cuda'] = torch.cuda.is_available()
    exp_config['ts'] = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))

    torch.manual_seed(exp_config['seed'])
    if exp_config['use_cuda']:
        torch.cuda.manual_seed_all(exp_config['seed'])
        
    if exp_config['logging']:
        exp_config['name'] = args.exp_name
        if not os.path.exists(exp_config["tb_logdir"] + "oracle_" + exp_config["name"]):
            os.makedirs(exp_config["tb_logdir"] + "oracle_" + exp_config["name"])
        writer = SummaryWriter(exp_config["tb_logdir"] + "oracle_" + exp_config["name"])
        train_batch_out = 0
        valid_batch_out = 0

    # Hyperparamters
    data_paths          = config['data_paths']
    optimizer_config    = config['optimizer']
    embedding_config    = config['embeddings']
    lstm_config         = config['lstm']
    mlp_config          = config['mlp']
    dataset_config      = config['dataset']
    inputs_config       = config['inputs']

    print("Loading MSCOCO bottomup index from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_index"]))
    with open(data_paths["FasterRCNN"]["mscoco_bottomup_index"]) as in_file:
        mscoco_bottomup_index = json.load(in_file)
        image_id2image_pos = mscoco_bottomup_index["image_id2image_pos"]
        image_pos2image_id = mscoco_bottomup_index["image_pos2image_id"]
        img_h = mscoco_bottomup_index["img_h"]
        img_w = mscoco_bottomup_index["img_w"]

    print("Loading MSCOCO bottomup features from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_features"]))
    mscoco_bottomup_features = np.load(data_paths["FasterRCNN"]["mscoco_bottomup_features"])

    print("Loading MSCOCO bottomup boxes from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_boxes"]))
    mscoco_bottomup_boxes = np.load(data_paths["FasterRCNN"]["mscoco_bottomup_boxes"])

    imgid2fasterRCNNfeatures = {}
    for mscoco_id, mscoco_pos in image_id2image_pos.items():
        imgid2fasterRCNNfeatures[mscoco_id] = dict()
        imgid2fasterRCNNfeatures[mscoco_id]["features"] = mscoco_bottomup_features[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["boxes"] = mscoco_bottomup_boxes[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_h"] = img_h[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_w"] = img_w[mscoco_pos]

    if dataset_config['new_vocab'] or not os.path.isfile(os.path.join(args.data_dir, data_paths['vocab_file'])):
        create_vocab(
            data_dir=args.data_dir,
            data_file=data_paths['train_file'],
            min_occ=dataset_config['min_occ'])

    with open(os.path.join(args.data_dir, data_paths['vocab_file'])) as file:
        vocab = json.load(file)

    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)

    lxmert_oracle_dict = {
        'no_words'            : vocab_size,
        'no_words_feat'       : embedding_config['no_words_feat'],
        'no_categories'       : embedding_config['no_categories'],
        'no_category_feat'    : embedding_config['no_category_feat'],
        'no_hidden_encoder'   : lstm_config['no_hidden_encoder'],
        'mlp_layer_sizes'     : mlp_config['layer_sizes'],
        'no_visual_feat'      : inputs_config['no_visual_feat'],
        'no_crop_feat'        : inputs_config['no_crop_feat'],
        'dropout'             : lstm_config['dropout'],
        'inputs_config'       : inputs_config,
        'scale_visual_to'     : inputs_config['scale_visual_to'],
        'lxmert_encoder_args' : inputs_config["LXRTEncoder"]
    }

    model = GEN(
        lxmert_oracle_dict = lxmert_oracle_dict,
        lxmert_oracle_weights = args.load_bin_path,
        pretrain_size = args.config
    )

    dataset_test = LXMERTOracleDataset(
        data_dir            = args.data_dir,
        data_file           = data_paths[args.set+'_file'],
        split               = args.set,
        visual_feat_file    = data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
        max_src_length      = dataset_config['max_src_length'],
        hdf5_visual_feat    = args.set+'_img_features',
        hdf5_crop_feat      = 'crop_features',
        imgid2fasterRCNNfeatures = imgid2fasterRCNNfeatures,
        history             = dataset_config['history'],
        new_oracle_data     = True, #dataset_config['new_oracle_data']
        successful_only     = dataset_config['successful_only'],
        load_crops=True,
        only_location=False
    )

    accuracy = []

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=exp_config['use_cuda']
    )

    torch.set_grad_enabled(False)
    model.eval()

    ans2tok = {'Yes': 1,
               'No': 0,
               'N/A': 2}

    tok2ans = {v: k for k, v in ans2tok.items()}

    qid2pos = {}

    encodings = np.zeros(shape=(len(dataloader), 768), dtype=np.float32)
    crosses = np.zeros(shape=(len(dataloader), 36, 768), dtype=np.float32)

    pos = 0
    newId = 0
    last_game_id = None
    stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    for i_batch, sample in stream:
        # Get Batch
        questions, answers, crop_features, visual_features, spatials, obj_categories, lengths = \
            sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], sample['obj_cat'], sample['length']
        # Forward pass
        history_raw = sample["history_raw"]
        fasterrcnn_features = sample['FasterRCNN']['features']
        fasterrcnn_boxes = sample['FasterRCNN']['boxes']
        pred_answer = model(Variable(questions),
            Variable(obj_categories),
            Variable(spatials),
            Variable(crop_features),
            Variable(visual_features),
            Variable(lengths),
            sample["history_raw"],
            sample['FasterRCNN']['features'],
            sample['FasterRCNN']['boxes'],
            sample["target_bbox"]
        )
        f = pred_answer[0][0].cpu().numpy()
        s = pred_answer[1][0].cpu().numpy()
        encodings[pos] = f
        crosses[pos] = s
        qid2pos[int(sample['qid'].numpy()[0])] = pos
        pos += 1
    with open(os.path.join(args.save_in, 'encoding_'+args.set+'.npy'), 'wb') as enc:
        np.save(enc, encodings)
    with open(os.path.join(args.save_in, 'crossAtt_'+args.set+'.npy'), 'wb') as cr:
        np.save(cr, crosses)
    with open(os.path.join(args.save_in, 'qid2pos_'+args.set+'.json'), 'w') as outfile:
        json.dump(qid2pos, outfile)
    