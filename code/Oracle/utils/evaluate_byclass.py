import json
from nltk.tokenize import TweetTokenizer
import copy

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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

    for key in game:
        questions = game[key]['qas']
        for que_idx, que in enumerate(questions):
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

    print('Attributes')
    for qtype in attributes:
        ques = attributes[qtype]
        qtype_count = len(ques)
        correct = sum([q['ans'] == q['model_ans'] for q in ques])
        print('{}: {:.2f}'.format(qtype, correct/qtype_count*100))

    print('Entities')
    for qtype in entities:
        ques = entities[qtype]
        qtype_count = len(ques)
        correct = sum([q['ans'] == q['model_ans'] for q in ques])
        print('{}: {:.2f}'.format(qtype, correct/qtype_count*100))

    correct = sum([q['ans'] == q['model_ans'] for q in na_type])
    print('NA: {:.2f}'.format(correct/len(na_type)*100))

    return (attributes, entities, na_type)

def compute_bycategory(filename="lxmert_scratch_small_predictions.csv"):


    print("Calculating Accuracy per category...")
    print("Getting DATA..")
    #TODO VARIABLE PATH
    data = pd.read_csv(os.path.join(os.getcwd(),filename))
    data = data.drop('Image',axis=1)
    #print(data.head())

    d = {}
    ID = -1
    push = {"qas" : []}
    for index, row in data.iterrows():
        if ID != row['Game ID']:
            if ID != -1:
                d[str(ID)] = push
                push = {"qas" : []}
            ID = row['Game ID']
        question = row["Question"].split('?')
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
        d[str(ID)]=push
    print("Done getting DATA!")
    print("Calculating...")
    count_and_rate(d)
