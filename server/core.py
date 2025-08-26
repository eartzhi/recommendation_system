import numpy as np 
import pandas as pd 
import pickle
import os
from pathlib import Path

from surprise import KNNWithMeans, KNNBasic, SVD
from surprise import Dataset
from surprise import Reader

base_dir = Path(__file__).resolve().parent
print(base_dir)

file_path_os = os.path.join(base_dir, "data", "xboost_items.pickle")
file_path_os_2 = os.path.join(base_dir, "data", "collab_model.pickle")

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

def restudy():
    try:
        events = pd.read_csv('data\events.csv.zip')
        events.drop_duplicates(inplace=True)
        
        item_properties_1 = pd.read_csv('data\item_properties_part1.csv')
        item_properties_2 = pd.read_csv('data\item_properties_part2.csv')
        item_properties =pd.concat([item_properties_1, item_properties_2], axis=0)
        item_properties = item_properties[item_properties['property']=='categoryid'][['itemid', 'value']]
        item_properties.drop_duplicates(inplace=True)
        
        item_properties['dupl'] = item_properties.itemid.duplicated()
        
        item_properties_1 = item_properties[item_properties['dupl']==True].reset_index(drop=True)
        item_properties_1['dupl'] = item_properties_1.itemid.duplicated() 
        item_properties_2 = item_properties_1[item_properties_1['dupl']==True].reset_index(drop=True)
        item_properties_2['dupl'] = item_properties_2.itemid.duplicated()
        item_properties_3 = item_properties_2[item_properties_2['dupl']==True].reset_index(drop=True)
        item_properties_3['dupl'] = item_properties_3.itemid.duplicated()
        
        items = item_properties[item_properties['dupl']==False].merge(item_properties_1[item_properties_1['dupl']==False], on='itemid', how='left', suffixes=('_0', '_1'))
        items = items.merge(item_properties_2[item_properties_2['dupl']==False], on='itemid', how='left', suffixes=('_1', '_2'))
        items = items.merge(item_properties_3[item_properties_3['dupl']==False], on='itemid', how='left', suffixes=('_2', '_3'))
        del item_properties, item_properties_1, item_properties_2, item_properties_3
        
        items =  items[['itemid', 'value_0', 'value_1', 'value_2', 'value_3']]
        items.reset_index(inplace=True, drop=True)
        items.value_0=items.value_0.astype(int)
        items.value_1=items.value_1.astype(int)
        items.value_2=items.value_2.astype(int)
        items.value_3=items.value_3.astype(int)
        
        items.fillna(-1, inplace=True)  
        category_tree = pd.read_csv('data\category_tree.csv')
        
        items_0 = items.merge(category_tree, left_on='value_0', right_on='categoryid', how='left', suffixes=('_0', '_1')).drop('categoryid', axis=1)
        items_1 = items_0.merge(category_tree, left_on='value_1', right_on='categoryid', how='left', suffixes=('_1', '_2')).drop('categoryid', axis=1)
        items_2 = items_1.merge(category_tree, left_on='value_1', right_on='categoryid', how='left', suffixes=('_2', '_3')).drop('categoryid', axis=1)
        items_3 = items_2.merge(category_tree, left_on='value_1', right_on='categoryid', how='left', suffixes=('_3', '_4')).drop('categoryid', axis=1)
        items = items_3
        
        del items_0, items_1, items_2, items_3
        
        items.rename(columns={'value_0': 'item_category_1',
                           'value_1': 'item_category_2',
                           'value_2': 'item_category_3',
                           'value_3': 'item_category_4',
                           'parentid_1': 'item_parent_category_1',
                           'parentid_2': 'item_parent_category_2',
                           'parentid_3': 'item_parent_category_3',
                           'parentid_4': 'item_parent_category_4'},
                  inplace=True)
        
        buy_with = events[events.event=='transaction'][['transactionid', 'itemid', 'timestamp']].groupby(['transactionid', 'itemid']).count().reset_index()
        buy_with.drop('timestamp', axis=1, inplace=True)
        
        buy_with['dupl'] = buy_with.transactionid.duplicated()
        buy_with_add = buy_with[buy_with['dupl']==True].groupby(['transactionid', 'itemid']).count().reset_index()
        buy_with_add['dupl'] = buy_with_add.transactionid.duplicated()
        
        buy_with_first = buy_with_add[buy_with_add['dupl']==False].groupby('transactionid').first()
        buy_with_first.drop('dupl', axis=1, inplace=True)
        buy_with_first.reset_index(inplace=True)
        
        buy_with_last = buy_with_add[buy_with_add['dupl']==True].groupby('transactionid').last()
        buy_with_last.drop('dupl', axis=1, inplace=True)
        buy_with_last.reset_index(inplace=True)
        
        buy_with = buy_with[buy_with['dupl']==False]
        buy_with = buy_with.merge(buy_with_first, on='transactionid', how='left', suffixes=('_0', '_1'))
        buy_with = buy_with.merge(buy_with_last, on='transactionid', how='left', suffixes=('_1', '_2'))
        buy_with.drop(['dupl', 'transactionid'], axis=1, inplace=True)
        buy_with.fillna(-1, inplace=True)
        
        buy_with.rename(columns={'itemid_0': 'itemid',
                           'itemid_1': 'buy_with_1',
                           'itemid': 'buy_with_2',},
                  inplace=True)
        
        items = items.merge(buy_with, on='itemid', how='left')
        items.fillna(-1, inplace=True)
        del buy_with
        
        events = events.merge(items, on='itemid', how='left')
        users = events[events.event=='view'][['visitorid', 'itemid', 'timestamp']].groupby(['visitorid','itemid']).count().reset_index()
        users = users.groupby('visitorid').max('timestamp').reset_index()[['visitorid', 'itemid']]
        
        users_cat = events[events.event=='view'][['visitorid', 'item_category_1', 
                                            'item_category_2', 'item_category_3', 
                                            'item_category_4', 'timestamp']].groupby(['visitorid', 'item_category_1', 
                                                                                      'item_category_2', 'item_category_3', 
                                                                                      'item_category_4'
                                                                                      ]).count().reset_index()
        users_cat = users_cat.groupby('visitorid').max('timestamp').reset_index()[['visitorid', 'item_category_1', 
                                                'item_category_2', 'item_category_3', 
                                                'item_category_4']]
        users = users.merge(users_cat, on='visitorid', how='left')
        del users_cat
        
        users_add_item = events[events.event=='addtocart'][['visitorid', 'itemid', 'timestamp']].groupby(['visitorid','itemid']).count().reset_index()
        users_add_item = users_add_item.groupby('visitorid').max('timestamp').reset_index()[['visitorid', 'itemid']]
        
        users = users.merge(users_add_item, on='visitorid', how='left', suffixes=('_viewed', '_added'))
        del users_add_item
        
        users_cat = events[events.event=='addtocart'][['visitorid', 'item_category_1', 
                                            'item_category_2', 'item_category_3', 
                                            'item_category_4', 'timestamp']].groupby(['visitorid', 'item_category_1', 
                                                                                      'item_category_2', 'item_category_3', 
                                                                                      'item_category_4'
                                                                                          ]).count().reset_index()
        users_cat = users_cat.groupby('visitorid').max('timestamp').reset_index()[['visitorid', 'item_category_1', 
                                                'item_category_2', 'item_category_3', 
                                                'item_category_4']]
        users = users.merge(users_cat, on='visitorid', how='left', suffixes=('_viewed', '_added'))
        del users_cat
        
        users_trans_item = events[events.event=='transaction'][['visitorid', 'itemid', 'timestamp']].groupby(['visitorid','itemid']).count().reset_index()
        users_trans_item = users_trans_item.groupby('visitorid').max('timestamp').reset_index()[['visitorid', 'itemid']]
        
        users = users.merge(users_trans_item, on='visitorid', how='left', suffixes=(None, '_buyed'))
        del users_trans_item
        
        users_cat = events[events.event=='transaction'][['visitorid', 'item_category_1', 
                                            'item_category_2', 'item_category_3', 
                                            'item_category_4', 'timestamp']].groupby(['visitorid', 'item_category_1', 
                                                                                      'item_category_2', 'item_category_3', 
                                                                                      'item_category_4'
                                                                                      ]).count().reset_index()
        users_cat = users_cat.groupby('visitorid').max('timestamp').reset_index()[['visitorid', 'item_category_1', 
                                                'item_category_2', 'item_category_3', 
                                                'item_category_4']]
        users = users.merge(users_cat, on='visitorid', how='left',)
    
        users.rename(columns={'itemid': 'itemid_buyed',
                               'item_category_1': 'item_category_1_buyed',
                               'item_category_2': 'item_category_2_buyed',
                               'item_category_3': 'item_category_3_buyed',
                               'item_category_4': 'item_category_4_buyed',
                               },
                                inplace=True)
        del users_cat
        
        daytime = events[['visitorid', 'time_of_day', 'timestamp'
                      ]].groupby(['visitorid', 'time_of_day']).count().reset_index()
        daytime.groupby('visitorid').max().reset_index()
        users = users.merge(daytime[['visitorid', 'time_of_day']], on='visitorid', how='left')
        
        daytime = events[['visitorid', 'dayofweek', 'timestamp'
                      ]].groupby(['visitorid', 'dayofweek']).count().reset_index()
        daytime.groupby('visitorid').max().reset_index()
        users = users.merge(daytime[['visitorid', 'dayofweek']], on='visitorid', how='left')
        users.rename(columns={'time_of_day': 'time_of_day_favorite',
                               'dayofweek': 'dayofweek_favorite'},
                      inplace=True)
        users.fillna(-1, inplace=True)
        del daytime
        
        events = events.merge(users, on='visitorid', how='left')
        
        return 'restudy completed'
    except BaseException:
        return 'error occured'