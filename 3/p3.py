#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('playtennis.csv')
print(dataset)

import math
def entropy(attributes):
  return sum([-attrib*math.log(attrib,2) for attrib in attributes])

from collections import Counter

def entropy_list(a_list):
  count = Counter(x for x in a_list)
  num_instances = len(a_list)*1.0
  attributes = [x/num_instances for x in count.values()]
  entropy_S = entropy(attributes)
  return entropy_S

def info_gain(df,split,target,trace=0):
    df_split = df.groupby(split)
    nobs = len(df.index)*1.0
    df_agg_ent = df_split.agg({ target:[entropy_list, lambda x: len(x)/nobs] })
    df_agg_ent.columns = ['Entropy','PropObserved']
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent["PropObserved"])
    old_entropy = entropy_list(df[target])
    return old_entropy - new_entropy

def id3(df,target,attribute_name,default_class = None):
    cnt = Counter(x for x in df[target])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or (not attribute_name):
        return default_class
    else:
        default_class = max(cnt.keys())
        gains = [info_gain(df,attr,target) for attr in attribute_name]
        index_max = gains.index(max(gains))
        best_attr = attribute_name[index_max]
        tree = { best_attr:{ } }
        remaining_attr = [x for x in attribute_name if x!=best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,target,remaining_attr,default_class)
            tree[best_attr][attr_val] = subtree
        return tree

def classify(instance,tree,default = None):
    attribute = next(iter(tree))
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result,dict):
            return classify(instance,result)
        else:
            return result
    else:
        return default

attribute_names = list(dataset.columns)
attribute_names.remove('play') #Remove the class attribute
tree = id3(dataset,'play',attribute_names)
print("\n\nThe Resultant Decision Tree is :\n")
print(tree)

training_data = dataset.iloc[1:-4] # all but last 4 instances and total 6
test_data = dataset.iloc[-4:] # just the last 4
train_tree = id3(training_data, 'play', attribute_names)
print("\n\nThe Resultant Decision train_tree is :\n")
print(train_tree)
test_data['predicted2'] = test_data.apply(classify,axis=1,args=(train_tree,'Yes') )
print ('\n\n Training the model for a few samples, and again predicting \'Playtennis\' for remaining attribute')
print('The Accuracy for new trained data is : ' + str( sum(test_data['play']==test_data['predicted2'] ) / (1.0*len(test_data.index)) ))
