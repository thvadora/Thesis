import os, sys
import json
import h5py
import gzip
import io
import copy
import numpy as np
import torch
import tqdm
from nltk.tokenize import TweetTokenizer
from utils.image_utils import get_spatial_feat
from torch.utils.data import Dataset
from transformers import BertTokenizer
import collections
from torch.utils.data import DataLoader

class DLXMERTOracleDataset(Dataset):
    def __init__(self, turns, data_path, sett, only_encodings):
        self.turns = turns
        self.data_path = data_path
        self.sett = sett
        self.only_encodings = only_encodings
        map_file = os.path.join(data_path, 'qid2pos_'+self.sett+'.json')
        with open(map_file) as json_file:
            self.qid2pos = json.load(json_file)
        encoding_file = os.path.join(data_path, 'encoding_'+self.sett+'.npy')
        self.encoding = np.load(encoding_file)
        if not self.only_encodings:
            cross_file = os.path.join(data_path, 'crossAtt_'+self.sett+'.npy')
            self.cross = np.load(cross_file)
        if self.sett == 'val':
            self.sett = 'valid'
        gw_file = os.path.join(data_path, 'guesswhat.'+self.sett+'.jsonl.gz')
        self.gw = []
        with gzip.open(gw_file) as file:
            for game in file:
                self.gw.append(json.loads(game.decode("utf-8")))
        
    def __len__(self):
        return len(self.gw)
    
    def __getitem__(self, idx):
        data = self.gw[idx]
        amount = self.turns
        dialog = data['qas']
        lxmertout = np.zeros(shape=(amount, 768), dtype=np.float32)
        for index, turn in enumerate(dialog):
            if index == amount:
                break
            qid = turn['id']
            position = self.qid2pos[str(qid)]
            lxmertout[index] = self.encoding[position]
        return lxmertout


if __name__ == '__main__':
    dataloader = DataLoader(
        dataset=DLXMERTOracleDataset(16, './data/', 'test', True),
        batch_size=2,
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=torch.cuda.is_available()
    )
    stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    for i_batch, sample in stream:
        print(sample.size())
        break
                
        
