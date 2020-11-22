import json
import gzip
import numpy as np
import matplotlib.pyplot as plt 

yes = 0
no = 0
with gzip.open('./data/guesswhat.valid.jsonl.gz') as file:
        for json_game in file:
            p = json.loads(json_game.decode("utf-8"))
            #if p['id'] in [2418, 2424, 2433]:
            #    print(p['id'], p['qas'])
            for q in p['qas']:
                ans = q['answer']
                yes += (ans=='Yes')
                no += (ans=='No')
print('Yes: ', yes)
print('No: ', no)

           