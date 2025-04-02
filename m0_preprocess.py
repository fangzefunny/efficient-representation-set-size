import os 
import pickle
import numpy as np 
import pandas as pd 

# set up path 
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists( f'{path}/data'): os.makedirs( f'{path}/data')

# utils
def clean_data(data):
    data = data.drop(columns = ['Unnamed: 0'])
    data = data.drop(index = data[(data == -1).values].index.unique().values)
    return data 

def remake_cols(data):
    header = ['subject', 'block', 'setSize', 'trial', 'state', 'img', 
              'imgCat', 'iter', 'correctAct', 'action', 'keyNum', 
              'correct', 'reward', 'RT', 'expCondi', 'pcor', 'delay']
    data.columns = header
    data.state -= 1
    data.action -= 1
    data.correctAct -= 1 
    return data 

def split_data(data, mode = 'block'):
    sub_lst = data['sub_id'].unique()
    train_data = {}

    if mode == 'subject':
        sub_lst = np.sort(data.sub_id.unique())
        for sub_id in sub_lst:
            train_data[sub_id] = data[ (data.sub_id==sub_id)]

    else:    
        for sub_id in sub_lst:
            sub2data = data.query(f'sub_id=={sub_id}')
            sub_data = {}
            block_list = np.sort(sub2data.block_id.unique())
            for block_id in block_list:
                sub3data = sub2data.query(f'block_id=={block_id}')
                xi = sub3data.loc[ :, [] ]
                sub_data[block_id] = xi
            train_data[sub_id] = sub_data

    return train_data 

def pre_process_12():
    
    # load data 
    human_data = pd.read_csv('raw_data/collins_12_orig.csv')
    human_data = clean_data( human_data)
    human_data = remake_cols(human_data)
    # rename the columns 
    human_data.rename(
        columns = {
            'setSize': 'block_type', 
            'state': 's', 
            'action': 'a', 
            'reward': 'r', 
            'iter': 'repetitions', 
            'correctAct': 'cor_a',
            'block': 'block_id',
            'subject': 'sub_id',
        }, inplace = True)
    human_data['sub_id'] = human_data['sub_id'].apply(
        lambda x: f'sub-{int(x):03d}'
    )
    
    # save the preprocessed data
    human_data['stage'] = 'train'
    human_data['a_ava'] = '[0,1,2]'
   
    # split data into subject data and save for fit 
    coi = ['sub_id', 'trial', 'block_id', 'block_type', 's', 'a', 'r', 
           'repetitions', 'cor_a', 'stage', 'a_ava']
    sub_lst = human_data['sub_id'].unique()
    data_for_fit = {}
    for_analysis = []
    for sub_id in sub_lst:
        sub_data = human_data.query(f'sub_id=="{sub_id}"')
        block_lst = sub_data['block_id'].unique()
        sub_data_for_fit = {}
        for block_id in block_lst:
            block_data = sub_data.query(f'block_id=={block_id}').reset_index(drop=True)
            block_data = block_data.loc[ :, coi]
            sub_data_for_fit[block_id] = block_data
            for_analysis.append(block_data)
        data_for_fit[sub_id] = sub_data_for_fit
    # save the for analysis data
    for_analysis = pd.concat(for_analysis)
    for_analysis.to_csv(f'{path}/data/setsize-collins12-human.csv')
    # save the preprocessed data
    with open(f'{path}/data/setsize-collins12.pkl', 'wb')as handle:
        pickle.dump(data_for_fit, handle)

def pre_process_14():
    
    # load data 
    human_data = pd.read_csv( 'raw_data/collins_14_orig.csv')
    human_data = clean_data( human_data)
    human_data = remake_cols( human_data)
    human_data = human_data.reset_index(drop=True)
     # rename the columns 
    human_data.rename(
        columns = {
            'setSize': 'block_type', 
            'state': 's', 
            'action': 'a', 
            'reward': 'r', 
            'iter': 'repetitions', 
            'correctAct': 'cor_a',
            'block': 'block_id',
            'subject': 'sub_id',
            'expCondi': 'group',
        }, inplace = True)
    human_data['sub_id'] = human_data['sub_id'].apply(lambda x: f'sub-{int(x)}')
    human_data['group'] = human_data['group'].apply(lambda x: 'HC' if x==0 else 'SZ')
    for group in ['HC', 'SZ']:
        n = human_data.query(f'group=="{group}"')["sub_id"].unique().__len__()
        print(f'{group}: {n}')

    # save the preprocessed data
    human_data['stage'] = 'train'
    human_data['a_ava'] = '[0,1,2]'
    human_data.to_csv( f'{path}/data/setsize-collins14-human.csv')

   # split data into subject data and save for fit 
    coi = ['sub_id', 'trial', 'block_id', 'block_type', 's', 'a', 'r', 
           'repetitions', 'cor_a', 'stage', 'a_ava', 'group']
    sub_lst = human_data['sub_id'].unique()
    data_for_fit = {}
    for_analysis = []
    for sub_id in sub_lst:
        sub_data = human_data.query(f'sub_id=="{sub_id}"')
        block_lst = sub_data['block_id'].unique()
        sub_data_for_fit = {}
        for block_id in block_lst:
            block_data = sub_data.query(f'block_id=={block_id}').reset_index(drop=True)
            block_data = block_data.loc[:, coi]
            sub_data_for_fit[block_id] = block_data
            for_analysis.append(block_data)
        data_for_fit[sub_id] = sub_data_for_fit
     # save the for analysis data
    for_analysis = pd.concat(for_analysis)
    for_analysis.to_csv(f'{path}/data/setsize-collins14-human.csv')
    # save the preprocessed data
    with open(f'{path}/data/setsize-collins14.pkl', 'wb')as handle:
        pickle.dump(data_for_fit, handle)
    

if __name__ == '__main__':
    
    # preprocess collins 12 data 
    pre_process_12()

    # preprocess collins 14 data 
    pre_process_14()
    

    

      
