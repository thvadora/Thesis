import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict
from time import time
import seaborn as sn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
import wget
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lxmert.src.lxrt.tokenization import BertTokenizer
from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget
from utils.config import load_config
from utils.datasets.Oracle.LXMERTOracleDataset import LXMERTOracleDataset
from utils.model_loading import load_model
from utils.vocab import create_vocab

use_cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="small", help='Config type [big, small]')
    parser.add_argument("-img_feat", type=str, default="rss", help='Select "vgg" or "res" as image features')
    parser.add_argument("-exp_name", type=str, default="exp", help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("-load_bin_path", type=str)
    parser.add_argument("-set", type=str, default="val")

    args = parser.parse_args()
    config_file = 'config/Oracle/config_small.json'
    
    if args.config == "big":
        config_file = 'config/Oracle/config.json'

    config = load_config(config_file)

    # Experiment Settings
    exp_config = config['exp_config']
    exp_config['img_feat'] = args.img_feat.lower()
    exp_config['use_cuda'] = use_cuda
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

    # Hyperparameters
    data_paths = config['data_paths']
    optimizer_config = config['optimizer']
    embedding_config = config['embeddings']
    lstm_config = config['lstm']
    mlp_config = config['mlp']
    dataset_config = config['dataset']
    inputs_config = config['inputs']

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

    ans2tok = {'Yes': 1,
               'No': 0,
               'N/A': 2}

    tok2ans = {v: k for k, v in ans2tok.items()}

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True
    )

    with open(os.path.join(args.data_dir, data_paths['vocab_file'])) as file:
        vocab = json.load(file)

    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)

    # Init Model, Loss Function and Optimizer
    model = LXMERTOracleInputTarget(
        no_words            = vocab_size,
        no_words_feat       = embedding_config['no_words_feat'],
        no_categories       = embedding_config['no_categories'],
        no_category_feat    = embedding_config['no_category_feat'],
        no_hidden_encoder   = lstm_config['no_hidden_encoder'],
        mlp_layer_sizes     = mlp_config['layer_sizes'],
        no_visual_feat      = inputs_config['no_visual_feat'],
        no_crop_feat        = inputs_config['no_crop_feat'],
        dropout             = lstm_config['dropout'],
        inputs_config       = inputs_config,
        scale_visual_to     = inputs_config['scale_visual_to'],
        lxmert_encoder_args = inputs_config["LXRTEncoder"]
    )
    model = load_model(model, args.load_bin_path, use_dataparallel=exp_config["use_cuda"])

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

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=exp_config['use_cuda']
    )

    wannasee = '68445'
    turn = 0

    stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)

    for i_batch, sample in stream:
        #print(dataset_test.oracle_data[idx])
        #sample = dataset_test.__getitem__(idx)
        if sample['game_id'][0] == wannasee:
            game_id = sample['game_id'][0]
            questions, answers, crop_features, visual_features, spatials, obj_categories, lengths = \
            sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], \
            sample['obj_cat'], sample['length']

            pred_answer = model(Variable(questions),
                 Variable(obj_categories),
                 Variable(spatials),
                 Variable(crop_features),
                 Variable(visual_features),
                 Variable(lengths),
                 sample["history_raw"],
                 sample['FasterRCNN']['features'],
                 sample['FasterRCNN']['boxes'],
                 sample["target_bbox"],
                 Variable(sample['train_features']['input_ids']),
                 Variable(sample['train_features']['input_mask']),
                 Variable(sample['train_features']['segment_ids'])
            )

            datapoint = 0
            
            try:
                img_fname = wget.download(sample['flickr'][datapoint])
            except:
                print('NO ESTA LA IMAGEN EN FLICKR')
                exit(1)

            os.rename(img_fname, os.path.join('./cachedimgs', sample['image_file'][datapoint]))
            im  = cv2.imread(os.path.join(os.path.join('./cachedimgs', sample['image_file'][datapoint])))
            im  = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            for layer in range(5):
                lang2vis_attention_probs = model.module.lxrt_encoder.model.bert.encoder.x_layers[layer].lang_att_map[datapoint].detach().cpu().numpy()

                vis2lang_attention_probs = model.module.lxrt_encoder.model.bert.encoder.x_layers[layer].visn_att_map[datapoint].detach().cpu().numpy()

                lang2vis_attention_probs = np.mean(lang2vis_attention_probs, axis=0)
                vis2lang_attention_probs = np.mean(vis2lang_attention_probs, axis=0)

                lang2vis_max_regions = np.argsort(lang2vis_attention_probs[0])[-5:][::-1]

                tokenized_history = tokenizer.tokenize(sample["history_raw"][datapoint])
                tokenized_history = ["<CLS>"] + tokenized_history + ["<SEP>"]
                lenofdialog = len(tokenized_history) 

                df_cm = pd.DataFrame(lang2vis_attention_probs[:lenofdialog], index = tokenized_history)

                fig = plt.figure(figsize=(30,6))

                plt.clf()

                ax = fig.add_subplot(111)
                ax.set_aspect(1)

                cmap = sn.cubehelix_palette(rot=-.4, light=1, as_cmap=True)

                res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=1.0, fmt='.2f', cmap=cmap, yticklabels=tokenized_history)

                plt.savefig('visualize_att/maps/game_'+str(game_id)+'_turn_'+str(turn)+'_layer'+str(layer)+'.png')

                plt.tight_layout()

                plt.close()

                plt.clf()

                plt.gca().set_axis_off()
            
                plt.imshow(im)

                for i, bbox in enumerate(sample["FasterRCNN"]["unnormalized_boxes"][datapoint][:35]):
                    if i in lang2vis_max_regions:
                        bbox = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]

                        if bbox[0] == 0:
                            bbox[0] = 2
                        if bbox[1] == 0:
                            bbox[1] = 2

                        plt.gca().add_patch(
                            plt.Rectangle((bbox[0], bbox[1]),
                                            bbox[2] - bbox[0] - 4,
                                            bbox[3] - bbox[1] - 4, fill=False,
                                            edgecolor='red', linewidth=2)
                        )
                        plt.text(bbox[0], bbox[1], str(i), color='b', fontsize=20)

                target_bbox = sample["unnormalized_target_bbox"][datapoint]

                plt.gca().add_patch(
                    plt.Rectangle((target_bbox[0], target_bbox[1]),
                                    target_bbox[2],
                                    target_bbox[3], fill=False,
                                    edgecolor='green', linewidth=2)
                )
                plt.text(target_bbox[0], target_bbox[1], '35', color='g', fontsize=20)

                tokenized_history = tokenizer.tokenize(sample["history_raw"][datapoint])
                tokenized_history = ["<CLS>"] + tokenized_history + ["<SEP>"]

                plt.tight_layout()
                plt.savefig(
                    "visualize_att/pics/lang2vis_game_{}_turn_{}_layer_{}_question_{}_answer_{}.png".format(
                        sample["game_id"][datapoint], turn, layer, " ".join(tokenized_history).replace("/", "_"),
                        tok2ans[sample["answer"][datapoint].item()].replace("/", "_")), bbox_inches='tight', pad_inches=0.5)
                plt.close()
                turn+=1


    
    # accuracy = []

    # dataloader = DataLoader(
    #     dataset=dataset_test,
    #     batch_size=optimizer_config['batch_size'],
    #     shuffle=False,
    #     num_workers=0 if sys.gettrace() else 4,
    #     pin_memory=exp_config['use_cuda']
    # )

    # torch.set_grad_enabled(False)
    # model.eval()

    # ans2tok = {'Yes': 1,
    #            'No': 0,
    #            'N/A': 2}

    # tok2ans = {v: k for k, v in ans2tok.items()}

    # tokenizer = BertTokenizer.from_pretrained(
    #     "bert-base-uncased",
    #     do_lower_case=True
    # )

    # plt.rcParams['figure.figsize'] = (12, 10)

    # annotations = {}
    # aux = pd.read_csv(os.path.join(args.data_dir, "locationq_classification_"+args.set+".csv"))
    # for index, row in aux.iterrows():
    #     annotations[(int(row['game_id']), int(row['dial_pos']))] = row

    # annotations_absolute = {}
    # aux = pd.read_csv(os.path.join(args.data_dir, "absolute_only_"+args.set+".csv"))
    # for index, row in aux.iterrows():
    #     annotations_absolute[(int(row['game_id']), int(row['dial_pos']))] = row

    # num_locations = 0
    # num_q = defaultdict(int)
    # selected_items = list(annotations.keys())

    # pos = 0
    # last_game_id = None
    # stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    # did = 0
    # for i_batch, sample in stream:
    #     questions, answers, crop_features, visual_features, spatials, obj_categories, lengths = \
    #         sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], \
    #         sample['obj_cat'], sample['length']

    #     # Forward pass
    #     pred_answer = model(Variable(questions),
    #             Variable(obj_categories),
    #             Variable(spatials),
    #             Variable(crop_features),
    #             Variable(visual_features),
    #             Variable(lengths),
    #             sample["history_raw"],
    #             sample['FasterRCNN']['features'], #problem of dimensions
    #             sample['FasterRCNN']['boxes'],
    #             sample["target_bbox"],
    #             Variable(sample['train_features']['input_ids']),
    #             Variable(sample['train_features']['input_mask']),
    #             Variable(sample['train_features']['segment_ids'])
    #         )

    #     if i_batch == 0:
    #         last_game_id = sample["game_id"][0]

    #     pred_answer_topk = pred_answer.topk(1)[1]

    #     for datapoint in range(len(sample["game_id"])):
    #         game_id = sample["game_id"][datapoint]
    #         if last_game_id != game_id:
    #             last_game_id = game_id
    #             pos = 0

    #         if (int(game_id), int(pos)) not in selected_items:
    #             pos += 1
    #             continue

    #         predictions_annotations = annotations[(int(game_id), int(pos))]
    #         relation_type = predictions_annotations['location_type']

    #         if predictions_annotations['cat'] != "['<spatial>']":
    #             pos += 1
    #             continue

    #         if relation_type == "absolute":
    #             absolute_annotations = annotations_absolute[(int(game_id), int(pos))]
    #             if absolute_annotations['is_true_absolute'] == "False":
    #                 pos += 1
    #                 continue

    #         num_q[relation_type] += 1
    #         num_locations += 1

    #         try:
    #             img_fname = wget.download(sample['flickr'][datapoint])
    #         except:
    #             pos += 1
    #             continue

    #         os.rename(img_fname, os.path.join('./cachedimgs', sample['image_file'][datapoint]))
    #         im = cv2.imread(
    #             os.path.join(os.path.join('./cachedimgs', sample['image_file'][datapoint])))
    #         im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    #         #print(model)
    #         for layer in range(5):
    #             lang2vis_attention_probs = model.module.lxrt_encoder.model.bert.encoder.x_layers[
    #                 layer].lang_att_map[datapoint].detach().cpu().numpy()

    #             vis2lang_attention_probs = model.module.lxrt_encoder.model.bert.encoder.x_layers[
    #                 layer].visn_att_map[datapoint].detach().cpu().numpy()

    #             lang2vis_attention_probs = np.mean(lang2vis_attention_probs, axis=0)
    #             vis2lang_attention_probs = np.mean(vis2lang_attention_probs, axis=0)

    #             lang2vis_max_regions = np.argsort(lang2vis_attention_probs[0])[-5:][::-1]

    #             tokenized_history = tokenizer.tokenize(sample["history_raw"][datapoint])
    #             tokenized_history = ["<CLS>"] + tokenized_history + ["<SEP>"]
    #             lenofdialog = len(tokenized_history) 

    #             df_cm = pd.DataFrame(lang2vis_attention_probs[:lenofdialog], index = tokenized_history)

    #             fig = plt.figure(figsize=(30,6))

    #             plt.clf()

    #             ax = fig.add_subplot(111)
    #             ax.set_aspect(1)

    #             cmap = sn.cubehelix_palette(rot=-.4, light=1, as_cmap=True)

    #             res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=1.0, fmt='.2f', cmap=cmap, yticklabels=tokenized_history)

    #             plt.savefig('att_maps/game_'+str(game_id)+'_turn_'+str(pos)+'_layer'+str(layer)+'.png')

    #             plt.tight_layout()

    #             plt.close()

    #             plt.clf()

    #             plt.gca().set_axis_off()
        
    #             plt.imshow(im)

    #             for i, bbox in enumerate(sample["FasterRCNN"]["unnormalized_boxes"][datapoint][:35]):
    #                 if i in lang2vis_max_regions:
    #                     bbox = [bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()]

    #                     if bbox[0] == 0:
    #                         bbox[0] = 2
    #                     if bbox[1] == 0:
    #                         bbox[1] = 2

    #                     plt.gca().add_patch(
    #                         plt.Rectangle((bbox[0], bbox[1]),
    #                                       bbox[2] - bbox[0] - 4,
    #                                       bbox[3] - bbox[1] - 4, fill=False,
    #                                       edgecolor='red', linewidth=2)
    #                     )
    #                     plt.text(bbox[0], bbox[1], str(i), color='b', fontsize=20)

    #             target_bbox = sample["unnormalized_target_bbox"][datapoint]

    #             plt.gca().add_patch(
    #                 plt.Rectangle((target_bbox[0], target_bbox[1]),
    #                               target_bbox[2],
    #                               target_bbox[3], fill=False,
    #                               edgecolor='green', linewidth=2)
    #             )
    #             plt.text(target_bbox[0], target_bbox[1], '35', color='g', fontsize=20)

    #             tokenized_history = tokenizer.tokenize(sample["history_raw"][datapoint])
    #             tokenized_history = ["<CLS>"] + tokenized_history + ["<SEP>"]

    #             plt.tight_layout()

    #             predictions_annotations = annotations[(int(game_id), int(pos))]
    #             relation_type = predictions_annotations['location_type']
    #             plt.savefig(
    #                 "att_visualizations/lang2vis_game_{}_turn_{}_layer_{}_type_{}_question_{}_answer_{}.png".format(
    #                     sample["game_id"][datapoint], pos, layer, relation_type, " ".join(tokenized_history).replace("/", "_"),
    #                     tok2ans[sample["answer"][datapoint].item()].replace("/", "_")), bbox_inches='tight', pad_inches=0.5)
    #             """plt.savefig(
    #                 "att_visualizations/lang2vis_game_{}_turn_{}_layer_{}_type_{}_question_{}_answer_{}.pdf".format(
    #                     sample["game_id"][datapoint], pos, layer, relation_type, " ".join(tokenized_history),
    #                     tok2ans[sample["answer"][datapoint].item()].replace("/", "_")), bbox_inches='tight', pad_inches=0.5)"""
    #             plt.close()
    #         pos += 1