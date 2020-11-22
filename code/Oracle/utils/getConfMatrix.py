import argparse
import csv
import datetime
import json
import sys
from time import time

import numpy as np
import pandas as pd
import os
import seaborn as sn
import math
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl

mpl.style.use('seaborn')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, help='CSV')
    parser.add_argument("-where", type=str, help="where to save it")
    args = parser.parse_args()

    data = pd.read_csv(args.data, keep_default_na=False)

    matrix = np.zeros(shape=(3,3), dtype=int)

    ans2tok = {
        'Yes' : 1,
        'No'  : 0,
        'N/A' : 2    
    }

    cyes = 0
    cno = 0
    cna = 0

    for index, row in data.iterrows():
        answer = ans2tok[str(row['GT Answer'])]
        pred = ans2tok[str(row['Model Answer'])]
        cyes += (answer==1)
        cno += (answer==0)
        cna += (answer==2)
        matrix[answer][pred] += 1

    sum = np.sum(matrix,axis=1)
    #print(sum)
    #matrix/=sum
    #print(matrix)
    #matrix[0] /= sum[0]
    #matrix[1] /= sum[1]
    #matrix[2] /= sum[2]

    #print(matrix)

    df_cm = pd.DataFrame(matrix, 
    index = [ 'No', 'Yes', 'N/A'],
    columns = [ 'No', 'Yes', 'N/A'])

    fig = plt.figure()

    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    cmap = sn.cubehelix_palette(light=1, as_cmap=True)

    res = sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=matrix.max(), fmt='.0f', cmap=cmap)

    plt.yticks([0.5,1.5,2.5], [ 'No', 'Yes', 'N/A'],va='center')

    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix_of_'+args.data[:-4]+'.png', dpi=100, bbox_inches='tight' )

    plt.close()
