import argparse
import csv
import datetime
import json
import sys
from time import time

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


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/Oracle/config_small.json", help='Config file')
    parser.add_argument("-img_feat", type=str, default="rss", help='Select "vgg" or "res" as image features')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("--preloaded", type=bool, default=False)

    args = parser.parse_args()

    config = load_config(args.config)

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
    mscoco_bottomup_features = None
    if args.preloaded:
        import sharearray
        print("Loading preloaded MS-COCO Bottom-Up features")
        mscoco_bottomup_features = sharearray.cache("mscoco_vectorized_features", lambda: None)
        mscoco_bottomup_features = np.array(mscoco_bottomup_features)
    else:
        mscoco_bottomup_features = np.load(data_paths["FasterRCNN"]["mscoco_bottomup_features"])

    print("Loading MSCOCO bottomup boxes from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_boxes"]))
    mscoco_bottomup_boxes = None
    if args.preloaded:
        print("Loading preloaded MS-COCO Bottom-Up boxes")
        mscoco_bottomup_boxes = sharearray.cache("mscoco_vectorized_boxes", lambda: None)
        mscoco_bottomup_boxes = np.array(mscoco_bottomup_boxes)
    else:
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

    print("Bert tokenizer or standard vocab?")

    with open(os.path.join(args.data_dir, data_paths['vocab_file'])) as file:
        vocab = json.load(file)

    # with open('./data/vocab_bert_tok.json') as file:
    #     vocab = json.load(file)

    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)

    dataset_test = LXMERTOracleDataset(
        data_dir            = args.data_dir,
        data_file           = data_paths['test_file'],
        split               = 'test',
        visual_feat_file    = data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
        max_src_length      = dataset_config['max_src_length'],
        hdf5_visual_feat    = 'test_img_features',
        hdf5_crop_feat      = 'crop_features',
        imgid2fasterRCNNfeatures = imgid2fasterRCNNfeatures,
        history             = dataset_config['history'],
        new_oracle_data     = True, #dataset_config['new_oracle_data']
        successful_only     = dataset_config['successful_only'],
        load_crops=True,
        only_location=False
    )

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=optimizer_config['batch_size'],
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=exp_config['use_cuda']
    )

    stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    k = 0
    for i_batch, sample in stream:

        questions, answers, crop_features, visual_features, spatials, obj_categories, lengths = \
                sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], sample['obj_cat'], sample['length']
        
        print(len(sample['FasterRCNN']['boxes']))
        k += 1
        if(k==3):
            break

test()