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

class POSDLXMERTOracleDataset(Dataset):
    def __init__(self, data_path, sett, onlyhist=False, succesful_only=True, onlyobj=False):
        self.data_path = data_path
        self.sett = sett
        self.onlyhist = onlyhist
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
        if self.sett == 'val':
            self.sett = 'valid'
        gw_file = os.path.join(data_path, 'guesswhat.'+self.sett+'.jsonl.gz')
        self.gw = []
        ans2tok = {
            'Yes' : 1,
            'No'  : 0,
            'N/A' : 2
        }
        with gzip.open(gw_file) as file:
            for game in file:
                g = json.loads(game.decode("utf-8"))
                if self.onlyhist:
                    if int(g['id']) not in self.historical:
                        continue
                if succesful_only:
                    if g['status'] == 'success':
                        encodings = []
                        for idx, q in enumerate(g['qas']):
                            res = {}
                            res['game_id'] = g['id']
                            res['question'] = q['question']
                            res['qid'] = q['id']
                            res['pos'] = idx
                            res['ans'] = ans2tok[q['answer']]
                            positionenc = self.qid2pos[str(q['id'])]
                            enc = self.encoding[positionenc]
                            res['encodings'] =  torch.tensor(encodings + [enc])
                            if q['answer'] == 'Yes':
                                if self.onlyobj:
                                    cat = classifier.que_classify_multi(q['question'].lower())
                                    if '<object>' in cat:
                                        encodings = encodings + [enc]
                                else:
                                    encodings = encodings + [enc]
                            self.gw.append(res)
                else:
                    encodings = []
                    for idx, q in enumerate(g['qas']):
                        res = {}
                        res['game_id'] = g['id']
                        res['question'] = q['question']
                        res['qid'] = q['id']
                        res['ans'] = torch.tensor(ans2tok[q['answer']])
                        res['pos'] = idx
                        positionenc = self.qid2pos[str(q['id'])]
                        enc = self.encoding[positionenc]
                        res['encodings'] =  torch.tensor(encodings + [enc])
                        if q['answer'] == 'Yes':
                            encodings = encodings + [enc]
                        self.gw.append(res)
    
    def __len__(self):
        return len(self.gw)
    
    def __getitem__(self, idx):
        return self.gw[idx]

if __name__ == '__main__':
    dataloader = DataLoader(
        dataset=POSDLXMERTOracleDataset('./data/', 'test'),
        batch_size=1,
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=torch.cuda.is_available()
    )
    stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
    for i_batch, sample in stream:
        print(sample)
        if i_batch > 30:
            break
                
        
