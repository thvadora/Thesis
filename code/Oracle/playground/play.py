import json
import gzip
import numpy as np
import matplotlib.pyplot as plt 

l = np.zeros(70)
n = 0

with gzip.open('./data/guesswhat.test.jsonl.gz') as file:
        for json_game in file:
            n += 1
            p = json.loads(json_game.decode("utf-8"))
            l[len(p['qas'])] += 1

res = sum(l)-sum(l[:17])
print('Cantidad de dialogos con mas de 10 preg: ', res)
print('Cantidad de dialogos que tengo en total: ', n)
print('Porcentaje que cubro con 16 turnos: ', 1-((res/n)))
for i,x in enumerate(l):
    if sum(l[i:])== 0:
        break
    print('Cantidad con ', i, 'turnos: ', x)
