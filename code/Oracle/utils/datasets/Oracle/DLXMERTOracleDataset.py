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
from question_analysis.qclassify import qclass
classifier = qclass()

class DLXMERTOracleDataset(Dataset):
    def __init__(self, turns, data_path, sett, only_encodings, succesful_only=True, onlyhist=False, onlypositives=False, onlyobj=False):
        self.turns = turns
        self.data_path = data_path
        self.sett = sett
        self.only_encodings = only_encodings
        self.onlyhist = onlyhist
        self.onlypositives = onlypositives
        self.onlyobj = onlyobj
        self.historical = []
        if self.onlyhist:
            file_name = os.path.join(data_path, 'ids_hist_dep.txt')
            f = open(file_name, 'r')
            for line in f.readlines():
                ide = int(line)
                self.historical.append(ide)
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
                g = json.loads(game.decode("utf-8"))
                if self.onlyhist:
                    if int(g['id']) not in self.historical:
                        continue
                if succesful_only:
                    if g['status'] == 'success':
                        self.gw.append(json.loads(game.decode("utf-8")))
                else:
                    self.gw.append(json.loads(game.decode("utf-8")))
        
    def __len__(self):
        return len(self.gw)
    
    def __getitem__(self, idx):
        data = self.gw[idx]
        dialog = data['qas']
        l = len(dialog)
        gameid = int(data['id'])
        #lxmertout = np.zeros(shape=(l, 768), dtype=np.float32)
        #ans = np.zeros(shape=(l), dtype=np.longlong)
        lxmertout = []
        ans = []
        ans2tok = {
            'Yes' : 1,
            'No'  : 0,
            'N/A' : 2
        }
        raws = []
        qids = []
        sz = 0
        for index, turn in enumerate(dialog):
            if self.onlypositives:
                if turn['answer'] == 'No':
                    continue
            if self.onlyobj:
                cat = classifier.que_classify_multi(turn['question'].lower())
                if '<object>' not in cat:
                    continue
            qid = turn['id']
            raws.append(turn['question'])
            qids.append(qid)
            position = self.qid2pos[str(qid)]
            lxmertout.append(self.encoding[position])
            ans.append(ans2tok[turn['answer']])
            sz += 1
        lxmertout = np.asarray(lxmertout)
        ans = np.asarray(ans)
        res = {
            'lxmertout' : torch.tensor(lxmertout),
            'sizes'     : sz,
            'answers'   : ans,
            'game_ids'  : gameid,
            'raw_q'     : raws,
            'qids'      : qids
        }
        return res
        #return (torch.tensor(lxmertout), len(dialog)), (ans, gameid)
        #return (torch.tensor([lxmertout[1]]), 1), (torch.tensor([ans[1]]), gameid)


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
                
        
