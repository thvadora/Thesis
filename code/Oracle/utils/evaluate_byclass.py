import json
from nltk.tokenize import TweetTokenizer
import copy

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import gzip
import argparse

from question_analysis.qclassify import qclass
classifier = qclass()

def split_que_ans2(game, human=False):
    new_game = copy.deepcopy(game)
    for key in game:
            new_game[key] = copy.deepcopy(game[key])
            qas = game[key]['qas']
            new_game[key]['que'] = []
            new_game[key]['ans'] = []
            if human:
                new_game[key]['mu'] = []
            for qa in qas:
                new_game[key]['que'].append(qa['question'])
                new_game[key]['ans'].append(qa['ans'])
                if human:
                    new_game[key]['mu'].append(qa['ans'] != qa['model_ans'])
    print('Splitting is done')
    return(new_game)

def count_and_rate(game, q_max_miss=0):

    tknzr = TweetTokenizer(preserve_case=False)
    q_count = 0

    entities = {'object': [], 'super-category': []}
    attributes = {'color': [],
                  'shape': [],
                  'size': [],
                  'texture': [],
                  'action': [],
                  'spatial': []}
    na_type = []

    ok = 0
    todo = 0
    for key in game:
        questions = game[key]['qas']
        for que_idx, que in enumerate(questions):
            ok += (que['ans']==que['model_ans'])
            todo+=1
            q_count +=1
            cat = '<NA>'
            cat = classifier.que_classify_multi(que['question'].lower())

            att_flag = False

            if '<color>' in cat:
                attributes['color'].append(que)
            if '<shape>' in cat:
                attributes['shape'].append(que)
            if '<size>' in cat:
                attributes['size'].append(que)
            if '<texture>' in cat:
                attributes['texture'].append(que)
            if '<action>' in cat:
                attributes['action'].append(que)
            if '<spatial>' in cat:
                attributes['spatial'].append(que)

            if '<object>' in cat:
                entities['object'].append(que)
            if '<super-category>' in cat:
                entities['super-category'].append(que)

            if cat == '<NA>':
                na_type.append(que)
    
    print('ACCURACY: ', ok/todo)

    print('Attributes')
    for qtype in attributes:
        ques = attributes[qtype]
        qtype_count = len(ques)
        correct = sum([q['ans'] == q['model_ans'] for q in ques])
        acc = 0
        try:
            acc = correct/qtype_count*100
        except:
            pass
        print('{}: {:.2f}'.format(qtype, acc), ' Number of question of this category: ', qtype_count)

    print('Entities')
    for qtype in entities:
        ques = entities[qtype]
        qtype_count = len(ques)
        correct = sum([q['ans'] == q['model_ans'] for q in ques])
        acc = 0
        try:
            acc = correct/qtype_count*100
        except:
            pass
        print('{}: {:.2f}'.format(qtype, acc), ' Number of question of this category: ', qtype_count)

    correct = sum([q['ans'] == q['model_ans'] for q in na_type])
    acc = 0
    try:
        acc = correct/qtype_count*100
    except:
        pass
    print('NA: {:.2f}'.format(acc), ' number of q : ', len(na_type))

    return (attributes, entities, na_type)

def compute_bycategory(filename, typee='val', onlyhist=False, namecol = 'kaka'):


    print("Calculating Accuracy per category...")
    print("Getting DATA..")
    #TODO VARIABLE PATH
    data = pd.read_csv(os.path.join(os.getcwd(),filename))
    dataset = pd.read_csv(os.path.join(os.getcwd(),filename))

    #savename = filename
    #historical = []
    #if onlyhist:
    #    file_name = os.path.join('data/', 'ids_hist_dep.txt')
    #    f = open(file_name, 'r')
    #    for line in f.readlines():
    #        historical.append(ide)
    #        ide = int(line)
    #    dataset = pd.read_csv(os.path.join(os.getcwd(), 'histdep.csv'))
    #    savename = 'histdep.csv'

    d = {}

    f = './data/guesswhat.valid.jsonl.gz'
    if typee == 'test':
        f = './data/guesswhat.test.jsonl.gz'
    tot = 0
    with gzip.open(f) as file:
        for json_game in file:
            game = json.loads(json_game.decode("utf-8"))
            #if onlyhist:
            #    if int(game['id']) not in historical:
            #        continue
            k = game['qas']
            #neededpositions = dataset.loc[dataset['Game ID']==game['id']]['Position'].to_list()
            if game['status'] == 'success':
                push = {'qas' : []}
                for index, q in enumerate(game['qas']):
                    #if onlyhist:
                    #    if index not in neededpositions:
                    #        continue
                    push['qas'].append({
                        "question": q['question'],
                        "ans": q['answer'],
                        "model_ans": data.loc[(data['Game ID']==game['id']) & (data['Position']==index)]['Model Answer'].item()
                    })
                    #ind = dataset.index[(dataset['Game ID']==game['id']) & (dataset['Position']==index)].to_list()[0]
                    #dataset.at[ind, namecol] = data.loc[(data['Game ID']==game['id']) & (data['Position']==index)]['Model Answer'].item()
                    tot+=1
                if push['qas'] != []:
                    d[str(game['id'])] = push
    #print(d)
    print('TOT: ', tot)
                
    """ID = -1
    push = {"qas" : []}
    for index, row in data.iterrows():
        if ID != row['Game ID']:
            if ID != -1:
                d[str(ID)] = push
                push = {"qas" : []}
            ID = row['Game ID']
        question = row["Input"].split('.')
        if question[-1] == "":
            question = question[-2]
        else:
            question = question[-1]
        push["qas"].append({
                "question":question,
                "ans":row["GT Answer"],
                "model_ans":row["Model Answer"]
                })
    if ID!="":
        d[str(ID)]=push"""


    #dataset.to_csv(savename)
    print("Done getting DATA!")
    print("Calculating...")
    count_and_rate(d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, help='CSV')
    parser.add_argument("-name", type=str, help="name")
    parser.add_argument("-is_historical", type=bool, help=False)
    args = parser.parse_args()

    compute_bycategory(args.data, 'test', args.is_historical, args.name)
