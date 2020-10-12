import json
import gzip
import numpy as np
import matplotlib.pyplot as plt 
from nltk.tokenize import TweetTokenizer


tknzr = TweetTokenizer(preserve_case=False)
#l = np.zeros(70)
n = 0
l = []
with gzip.open('./data/guesswhat.train.jsonl.gz') as file:
        for json_game in file:
            n += 1
            p = json.loads(json_game.decode("utf-8"))
            k = p['qas']
            res = 0
            meti = 0
            if p['status'] == 'success':
                for x in k:
                    q_tokens = tknzr.tokenize(x['question'])
                    """if meti == 0:
                        res += len(q_tokens)
                        meti = 1
                    elif x['answer'] == 'Yes':
                        res += len(q_tokens)"""
                    res += len(q_tokens)
                l.append(res)

a = np.array(l)
p = np.percentile(a, 99)
# An "interface" to matplotlib.axes.Axes.hist() method
fig, ax = plt.subplots()
n, bins, patches = plt.hist(x=l, bins='auto', color='#0504aa',
                            alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Tamaño de los diálogos en tokens')
plt.ylabel('Cantidad de díalogos')
plt.title('Histograma tamaño de dialogos (all, train set)')
txt = "Promedio de tokens por dialogo " + str(sum(l)/len(l))+ "\n Para cubrir el 99% necesitamos considerar "+ str(p)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
ax.set_xlim([0,150])
plt.savefig('all.png')
#print(l)
"""res = sum(l)-sum(l[:17])
print('Cantidad de dialogos con mas de 10 preg: ', res)
print('Cantidad de dialogos que tengo en total: ', n)
print('Porcentaje que cubro con 16 turnos: ', 1-((res/n)))
for i,x in enumerate(l):
    if sum(l[i:])== 0:
        break
    print('Cantidad con ', i, 'turnos: ', x)"""
