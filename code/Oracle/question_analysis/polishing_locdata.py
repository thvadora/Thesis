import pandas as pd

# Run this to filter out absolute questions with more than
# <spatial> in their category
#locdata = pd.read_csv('locationq_classification.csv')
#locdata['cat'] = locdata['cat'].astype(str)
#cond = ~(locdata['location_type'] == 'absolute') | (locdata['cat'] == "['<spatial>']")
#newlocdata = locdata[cond]
#newlocdata.to_csv('locdata_with_absolute_filtered.csv', index=False)

newlocdata = pd.read_csv('./locdata_with_absolute_filtered.csv')
# Run this to classify the remaining absolute questions between
# true absolute and non true absolute (kinda weird name)
absoluteonly = newlocdata[ newlocdata['location_type'] == 'absolute']

superlatives = """rightmost 
 leftmost 
 nearest 
 closest 
 most right 
 left most 
 most left 
 right most 
 extreme right 
 extreme left""" 
superlatives = superlatives.split('\n') 

rest = """front 
 background 
 left 
 middle 
 foreground 
 right 
 top 
 backmost 
 near 
 bottom 
 far 
 back 
 closer 
 center 
 picture 
 side 
 half 
 corner 
 image""" 
rest = rest.split('\n')

def is_true_absolute(q):
    words = q.split()
    end_pos = len(words)
    for i, w in enumerate(words):
        if w in superlatives:
            return True
        if w in rest and i < end_pos:
            return True
    return False

absoluteonly['is_true_absolute'] = absoluteonly['question'].map(lambda x: is_true_absolute(x))
absoluteonly.to_csv('absolute_spatial_only.csv', index=False)
