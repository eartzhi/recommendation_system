import numpy as np 
import pandas as pd 
import pickle
import os
from pathlib import Path

from surprise import KNNWithMeans, KNNBasic, SVD
from surprise import Dataset
from surprise import Reader

file_path_os = os.path.join(os.getcwd(), "data", "xboost_items.pickle")
file_path_os_2 = os.path.join(os.getcwd(), "data", "collab_model.pickle")

with open(file_path_os, 'rb') as f:
    items = pickle.load(f)
    
with open(file_path_os_2, 'rb') as f:
    model = pickle.load(f)
    
# with open('data\xboost_items.pickle', 'rb', encoding="utf-8") as f:
#     items = pickle.load(f)
    
def matching(user, model=model):
    users_list =[]
    items_list =[]
    score_list =[]
    for it in items.itemid.unique():
        prediction = model.predict(user, it)
        users_list.append(prediction[0])
        items_list.append(prediction[1])
        score_list.append(prediction[3])
    return pd.DataFrame({'visitorid': users_list, 'itemid': items_list, 'score': score_list}).nlargest(3, 'score')