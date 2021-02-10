import json
from nltk.tokenize import TweetTokenizer
import copy

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gzip
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-models", type=str, help='coma separated models names')
    args = parser.parse_args()

    l = args.models.split(',')

    full = pd.read_csv('histdep2.csv', keep_default_na=False, index_col=[0])
    full = full.loc[full["inverse answer without history"]=='1']
    for model in l:
        filename = model+'testpredictions.csv'
        data = pd.read_csv(os.path.join(os.getcwd(),filename), keep_default_na=False)
        data = data.reset_index()
        data = data.rename(columns={"Model Answer" : model})
        
        full = pd.merge(full, data[[model, 'Game ID', 'Position']], how='left', on=['Game ID', 'Position'])

        #calculate accuracies
        types = ['history', 'middle', 'x-axis', 'y-axis']
        typetots = np.zeros(len(types))
        typeoks = np.zeros(len(types))
        tot = 0
        ok = 0
        for index, row in full.iterrows():
            tot += 1
            ok += (row['GT Answer'].lower() == row[model].lower())
            for idx, t in enumerate(types):
                if row['error-type'] == t:
                    typetots[idx] += 1
                    typeoks[idx] += (row['GT Answer'].lower() == row[model].lower())

        print('MODEL: ', model)
        print('ACCURACY FOR THE MINI TEST SET IS : ', ok/tot, ' amount of q is: ', tot)
        for idx, t in enumerate(types):
            print('ACURACCY FOR TYPE ', t, ': ', typeoks[idx]/typetots[idx], 'FORMULA: ', typeoks[idx], '/', typetots[idx])



    full.to_csv('minitestpredictions.csv')