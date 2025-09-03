import numpy as np 
import pandas as pd 
import pickle
import os
from pathlib import Path

from surprise import SVD
from surprise import Dataset
from surprise import Reader

base_dir = Path(__file__).resolve().parent


error_counter = 0

items_path = os.path.join(base_dir, "data", "items_unique.pickle")
users_path = os.path.join(base_dir, "data", "items_unique.pickle")
model_path = os.path.join(base_dir, "data", "matrix_model.pickle")
top_products_path = os.path.join(base_dir, "data", "top_3_products.pickle")

with open(items_path, 'rb') as f:
    items = pickle.load(f)
    
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
with open(users_path, 'rb') as f:
    users = pickle.load(f)

with open(top_products_path, 'rb') as f:
    top_products = pickle.load(f)

    
def matching(user, model=model):
    users_list =[]
    items_list =[]
    score_list =[]
    for it in items:
        prediction = model.predict(user, it)
        users_list.append(prediction[0])
        items_list.append(prediction[1])
        score_list.append(prediction[3])
    return pd.DataFrame({'visitorid': users_list, 'itemid': items_list, 'score': score_list}).nlargest(3, 'score')


def relearning(parameters):
    global model
    try:
        datapath = os.path.join(base_dir, "data", "matrix_dataset.pickle")
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
            
        trainset = data.build_full_trainset()
        
        new_model = SVD(**parameters)
        new_model.fit(trainset)
        model = new_model
        return 'relearning completed'
    
    except BaseException as err:
        error_counter += 1
        return str(err)


def reassemble():
    global error_counter
    try:
        events_path = os.path.join(base_dir, "data", "events.csv.zip")
        events = pd.read_csv(events_path)
        events.drop_duplicates(inplace=True)
        events['datetime'] = pd.to_datetime(events.timestamp, unit='ms')
        
        daytime = {'morning':list(range(6,12)), 'afternoon':list(range(12,18)), 'evening':list(range(18,23))}
        
        def time_of_day(x):
            for i in daytime.keys():
                if x in daytime[i]:
                    return i
            return 'night'
        
        events['month'] = events['datetime'].dt.month
        events['dayofweek'] = events['datetime'].dt.dayofweek
        
        events['time_of_day'] = events['datetime'].dt.hour
        events['time_of_day'] = events['time_of_day'].apply(time_of_day)        
                
        item_properties_1_path = os.path.join(base_dir, "data", "item_properties_part1.csv")
        item_properties_2_path = os.path.join(base_dir, "data", "item_properties_part2.csv")
        item_properties_1 = pd.read_csv(item_properties_1_path)
        item_properties_2 = pd.read_csv(item_properties_2_path)
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
        items.reset_index(inplace=True, drop=True)
        items.fillna(-1, inplace=True)
        del item_properties, item_properties_1, item_properties_2, item_properties_3
        
        items =  items[['itemid', 'value_0', 'value_1', 'value_2', 'value_3']]
        items.reset_index(inplace=True, drop=True)
        items.value_0=items.value_0.astype(int)
        items.value_1=items.value_1.astype(int)
        items.value_2=items.value_2.astype(int)
        items.value_3=items.value_3.astype(int)
        
        category_tree_path = os.path.join(base_dir, "data", "category_tree.csv") 
        category_tree = pd.read_csv(category_tree_path)
        
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
        
        event_dict = {'view':1, 'addtocart':5,'transaction':10}
        cat_coef = 0.1
        parent_cat_coef = 0.01
        
        events.event = events.event.apply(lambda x: event_dict[x])
        rating = events[['visitorid', 'itemid', 'event']].groupby(['visitorid', 'itemid']).mean().reset_index()
        rating.rename(columns={'event': 'item_rating'}, inplace=True)
        
        cat_added_rating  = pd.lreshape(events[['visitorid', 'item_category_1_added', 
                                  'item_category_2_added', 'item_category_3_added', 
                                  'item_category_4_added']], {'item_category': ['item_category_1_added', 'item_category_2_added', 
                                                                       'item_category_3_added', 'item_category_4_added']})
        cat_added_rating = cat_added_rating[cat_added_rating.item_category!=-1]
        cat_added_rating.drop_duplicates(inplace=True)
        # cat_added_rating.dropna(how='any', inplace=True)
        cat_added_rating = cat_added_rating.astype('int32')
        
        # Купленые товары
        cat_buyed_rating  = pd.lreshape(events[['visitorid', 'item_category_1_buyed', 
                                          'item_category_2_buyed', 'item_category_3_buyed', 
                                          'item_category_4_buyed']], {'item_category': ['item_category_1_buyed', 'item_category_2_buyed', 
                                                                               'item_category_3_buyed', 'item_category_4_buyed']})
        cat_buyed_rating = cat_buyed_rating[cat_buyed_rating.item_category!=-1]
        cat_buyed_rating.drop_duplicates(inplace=True)
        # cat_buyed_rating.dropna(how='any', inplace=True)
        cat_buyed_rating = cat_buyed_rating.astype('int32')
        
        items_cat = pd.lreshape(items[['itemid', 'item_category_1', 
                                  'item_category_2', 'item_category_3', 
                                  'item_category_4']], {'item_category': ['item_category_1', 'item_category_2', 
                                                                       'item_category_3', 'item_category_4']})
        items_cat = items_cat[items_cat.item_category!=-1] 
        items_cat = items_cat.astype('int32')
        
        cat_added_rating = cat_added_rating.merge(items_cat, on='item_category', how='outer')
        cat_added_rating.dropna(inplace=True)
        cat_added_rating.drop('item_category', axis=1, inplace=True)
        cat_added_rating.drop_duplicates(inplace=True)
        cat_added_rating['cat_rating_added'] =cat_coef * 5
        
        cat_buyed_rating = cat_buyed_rating.merge(items_cat, on='item_category', how='outer')
        cat_buyed_rating.dropna(inplace=True)
        cat_buyed_rating.drop('item_category', axis=1, inplace=True)
        cat_buyed_rating.drop_duplicates(inplace=True)
        cat_buyed_rating['cat_rating_buyed'] =cat_coef * 10
        
        rating = rating.merge(cat_added_rating, on=['itemid', 'visitorid'], how='outer')
        rating = rating.merge(cat_buyed_rating, on=['itemid', 'visitorid'], how='outer')
        rating = rating.groupby(['visitorid', 'itemid']).mean().reset_index()
        del items_cat
        
        # Добавленые в корзину товары
        parent_added_rating = cat_added_rating[['visitorid', 'itemid']]
        parent_added_rating.drop_duplicates(inplace=True)
        
        # Купленые товары
        parent_buyed_rating = cat_buyed_rating[['visitorid', 'itemid']]
        parent_buyed_rating.drop_duplicates(inplace=True)
        del cat_added_rating, cat_buyed_rating
        
        # Родительские категории товаров
        items_parent_cat  = pd.lreshape(items[['itemid', 'item_parent_category_1', 
                                          'item_parent_category_2', 'item_parent_category_3', 
                                          'item_parent_category_4']], {'item_parent_category': ['item_parent_category_1', 'item_parent_category_2', 
                                                                               'item_parent_category_3', 'item_parent_category_4']})
        items_parent_cat.drop_duplicates(inplace=True)
        # cat_added_rating.dropna(how='any', inplace=True)
        items_parent_cat = items_parent_cat[items_parent_cat.item_parent_category!=-1] 
        items_parent_cat = items_parent_cat.astype('int32')
                
        parent_added_rating = parent_added_rating.merge(items_parent_cat, on='itemid', how='outer')
        parent_added_rating = parent_added_rating[['visitorid', 'item_parent_category']].drop_duplicates()
        parent_buyed_rating = parent_buyed_rating.merge(items_parent_cat, on='itemid', how='outer')
        parent_buyed_rating = parent_buyed_rating[['visitorid', 'item_parent_category']].drop_duplicates()
        
        parent_added_rating = parent_added_rating.merge(items_parent_cat, on='item_parent_category', how='outer')
        parent_added_rating.dropna(inplace=True)
        parent_added_rating.drop('item_parent_category', axis=1, inplace=True)
        parent_added_rating.drop_duplicates(inplace=True)
        parent_added_rating['parent_cat_rating_added'] =parent_cat_coef * 5
        
        parent_buyed_rating = parent_buyed_rating.merge(items_parent_cat, on='item_parent_category', how='outer')
        parent_buyed_rating.dropna(inplace=True)
        parent_buyed_rating.drop('item_parent_category', axis=1, inplace=True)
        parent_buyed_rating.drop_duplicates(inplace=True)
        parent_buyed_rating['parent_cat_rating_buyed'] =parent_cat_coef * 10
        
        rating = rating.merge(parent_added_rating, on=['itemid', 'visitorid'], how='left')
        rating = rating.merge(parent_buyed_rating, on=['itemid', 'visitorid'], how='left')
        rating = rating.groupby(['visitorid', 'itemid']).mean().reset_index()
        
        rating = rating.fillna(0)
        rating['rating_total'] = rating.item_rating + rating.cat_rating_added \
        + rating.cat_rating_buyed + rating.parent_cat_rating_added + rating.parent_cat_rating_buyed
        rating.drop(['item_rating', 'cat_rating_added', 'cat_rating_buyed', 'parent_cat_rating_added', 'parent_cat_rating_buyed'], axis=1, inplace=True)
        del parent_added_rating, parent_buyed_rating
            
            
        dataset = rating.rename(columns={
            "visitorid": "userID",
            "itemid": "itemID", 
            'rating_total':'rating'
        })   
        
        reader = Reader(rating_scale=(rating.rating_total.min(), rating.rating_total.max()))
        data = Dataset.load_from_df(dataset, reader)
        trainset = data.build_full_trainset()
                        
                
        return 'restudy completed'
    except BaseException as err:
        error_counter += 1
        return str(err)