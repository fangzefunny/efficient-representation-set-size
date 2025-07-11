import os 
import argparse
import pickle
import numpy as np 
import pandas as pd 

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',   '-d', help='which_data', type=str, default='setsize-cluster')
args = parser.parse_args()

# set up path 
pth = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists( f'{pth}/data'): os.makedirs( f'{pth}/data')

# utils
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
    human_data = pd.read_csv('raw_data/collins_12_orig.csv', index_col=0)
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
            accept = True
            block_data = sub_data.query(f'block_id=={block_id}').reset_index(drop=True)
            block_data = block_data.loc[ :, coi]
            # preprocess the data with nan value 
            n_nan = block_data.query('r==-1').shape[0]
            if n_nan > 0:
                thres = np.max([block_data.shape[0] * .1, 3])
                accept = False if n_nan > thres else True
                if accept:
                    # fill the a==-2 with a random action from [0, 1, 2]
                    nan_idx = block_data.query('r==-1').index
                    block_data.loc[nan_idx, 'a'] = np.random.choice(3, size=n_nan)
                    block_data.loc[block_data.query('r==-1').index, 'r'] = (
                        block_data.query('r==-1')['a'] == 
                        block_data.query('r==-1')['cor_a']
                    ).astype(int)
                else:
                    print(f'{sub_id}, block-{int(block_id)} has {n_nan} nan values')
            if accept:
                sub_data_for_fit[block_id] = block_data
                for_analysis.append(block_data)
        data_for_fit[sub_id] = sub_data_for_fit
    # save the for analysis data
    for_analysis = pd.concat(for_analysis)
    for_analysis.to_csv(f'{pth}/data/setsize-collins12-human.csv')
    # save the preprocessed data
    with open(f'{pth}/data/setsize-collins12.pkl', 'wb')as handle:
        pickle.dump(data_for_fit, handle)

def pre_process_14():
    
    # load data 
    human_data = pd.read_csv('raw_data/collins_14_orig.csv', index_col=0)
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
    human_data.to_csv( f'{pth}/data/setsize-collins14-human.csv')

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
            accept = True
            block_data = sub_data.query(f'block_id=={block_id}').reset_index(drop=True)
            block_data = block_data.loc[ :, coi]
            # preprocess the data with nan value 
            n_nan = block_data.query('r==-1').shape[0]
            if n_nan > 0:
                thres = np.max([block_data.shape[0] * .1, 3])
                accept = False if n_nan > thres else True
                if accept:
                    # fill the a==-2 with a random action from [0, 1, 2]
                    nan_idx = block_data.query('r==-1').index
                    block_data.loc[nan_idx, 'a'] = np.random.choice(3, size=n_nan)
                    # block_data.loc[block_data.query('r==-1').index, 'r'] = (
                    #     block_data.query('r==-1')['a'] == 
                    #     block_data.query('r==-1')['cor_a']
                    # ).astype(int)
                else:
                    print(f'{sub_id}, block-{int(block_id)} has {n_nan} nan values')
            if accept:
                sub_data_for_fit[block_id] = block_data
                for_analysis.append(block_data)
        data_for_fit[sub_id] = sub_data_for_fit
    # save the for analysis data
    for_analysis = pd.concat(for_analysis)
    for_analysis.to_csv(f'{pth}/data/setsize-collins14-human.csv')
    # save the preprocessed data
    with open(f'{pth}/data/setsize-collins14.pkl', 'wb')as handle:
        pickle.dump(data_for_fit, handle)
    
def pre_process_uncomp(data_set):

    # columns of interest
    coi = ['sub_id', 'trials', 'block_id', 'block_type', 's', 'a', 'r', 
        'repetitions', 'cor_a', 'stage', 'a_ava']
    
    ## Loop to preprocess each file
    # obtain all files under the exp1 list
    files = os.listdir(f'{pth}/raw_data/{data_set}/')
    n_sub = 0
    data_for_fit = {}
    for_analysis = []
    for file in files:
        # skip the folder 
        if file.endswith('.csv'):
            n_sub += 1
            # get and remake the cols and index
            sub_data = pd.read_csv(f'{pth}/raw_data/{data_set}/{file}', encoding='latin1')
            if "screen_id" in sub_data.columns:
                sub_data = sub_data.query('screen_id=="sub_response"').reset_index(drop=True)
            # remove the pratice trials
            sub_data = sub_data.query('pic_typ!="prac"').reset_index(drop=True)
            # rename columns
            sub_data.rename(
                columns = {
                    'block_typ': 'block_type',
                    'a_cor': 'cor_a',
                    'tps': 'repetitions',
                    'trial': 'trials',                    
                }, inplace = True)
            # split the data into blocks
            sub_data['block_id'] = sub_data['block_id'] - 1
            block_lst = sub_data['block_id'].unique()
            # get the sub_id
            #sub_id = file.split('.')[0]
            sub_id = file.split('_')[2]
            sub_data_for_fit = {}
            for block_id in block_lst:
                block_data = sub_data.query(f'block_id=={block_id}').reset_index(drop=True)
                block_data['sub_id'] = f'sub-{sub_id}'
                # make sure the stimulus has range from 0-7
                s = block_data['s'].unique()
                assert np.min(s)==0 and np.max(s)<=9, f's has value other than 0-7'
                # make sure the action has range from 0-7, -1 is not response 
                block_data['a'] = block_data['a']
                #assert np.min(block_data['a'])>=0 and np.max(block_data['a'])<=9, f'a has value other than 0-7'
                # ensure the repetitions has range from 1-30
                block_data['repetitions'] = block_data['repetitions'].apply(lambda x: x+1)
                assert np.min(block_data['repetitions'])>=1 and np.max(block_data['repetitions'])<=30, f'repetitions has value other than 1-30'
                # find the not reponse trials
                nan_idx = block_data.query('a==999').index
                n_nan = nan_idx.shape[0]
                accept = True
                if n_nan > 0:
                    thres = np.max([block_data.shape[0] * .05, 3])
                    accept = False if n_nan > thres else True
                    if accept:
                        print(f'sub-{sub_id}, block-{int(block_id)} has {n_nan} nan values, accept: {accept}')
                        # fill the nan row with a random action from 
                        # get action space
                        nA = int(block_data['block_type'].unique()[0].split('S')[1].split('A')[0])
                        block_data.loc[nan_idx, 'a'] = np.random.choice(nA, size=n_nan)
                        # update the reward 
                        block_data.loc[nan_idx, 'r'] = (
                            block_data.loc[nan_idx, 'a'] == 
                            block_data.loc[nan_idx, 'cor_a']
                        ).astype(int)
                    else:
                        print(f'sub-{sub_id}, block-{int(block_id)} has {n_nan} nan values, accept: {accept}')

                if accept:
                    sub_data_for_fit[block_id] = block_data[coi]
                    for_analysis.append(block_data[coi])

            # save the data for fit
            data_for_fit[sub_id] = sub_data_for_fit
                
    # save the for analysis data
    for_analysis = pd.concat(for_analysis)
    for_analysis.to_csv(f'{pth}/data/{data_set}-human.csv')
    # save the preprocessed data
    with open(f'{pth}/data/{data_set}.pkl', 'wb')as handle:
        pickle.dump(data_for_fit, handle)

def pre_process_cluster(data_set):

    # columns of interest
    coi = ['sub_id', 'trials', 'block_id', 'block_type', 's', 'a', 'r', 
        'repetitions', 'cor_a', 'stage', 'a_ava', 'pair_type']
    
    ## Loop to preprocess each file
    # obtain all files under the exp1 list
    files = os.listdir(f'{pth}/raw_data/{data_set}/')
    n_sub = 0
    data_for_fit = {}
    for_analysis = []
    memory_data  = []
    for file in files:
        # skip the folder 
        if file.endswith('.csv'):
            n_sub += 1
            # get and remake the cols and index
            sub_data = pd.read_csv(f'{pth}/raw_data/{data_set}/{file}', encoding='latin1')
            # remove the pratice trials
            if "screen_id" in sub_data.columns:
                sub_data = sub_data.query('screen_id=="sub_response"').reset_index(drop=True)
            sub_data = sub_data.query('pic_typ!="prac"').reset_index(drop=True)
            # rename columns
            sub_data.rename(
                columns = {
                    'block_typ': 'block_type',
                    'a_cor': 'cor_a',
                    'tps': 'repetitions',
                    'trial': 'trials',                    
                }, inplace = True)
            sub_data['block_id'] = sub_data['block_id'] - 1
            # split the data according to its phase
            learn_data = sub_data.query('phase=="learn"').reset_index(drop=True)
            mem_data = sub_data.query('phase!="learn"').reset_index(drop=True)
            # get the sub_id
            sub_id = file #.split('_')[2].split('-')[1]

            # ----------------- preprocess the learning data ----------------- #
            sub_data_for_fit = {}
            block_lst = learn_data['block_id'].unique()
            for block_id in block_lst:
                block_data = learn_data.query(f'block_id=={block_id}').reset_index(drop=True)
                block_data['sub_id'] = f'sub-{sub_id}'
                # make sure the stimulus has range from 0-7
                s = block_data['s'].unique()
                assert np.min(s)==0 and np.max(s)<=9, f's has value other than 0-7'
                # make sure the action has range from 0-7, -1 is not response 
                block_data['a'] = block_data['a']
                #assert np.min(block_data['a'])>=0 and np.max(block_data['a'])<=9, f'a has value other than 0-7'
                # ensure the repetitions has range from 1-30
                block_data['repetitions'] = block_data['repetitions'].apply(lambda x: x+1)
                assert np.min(block_data['repetitions'])>=1 and np.max(block_data['repetitions'])<=30, f'repetitions has value other than 1-30'
                # find the not reponse trials
                nan_idx = block_data.query('a==999').index
                n_nan = nan_idx.shape[0]
                accept = True
                if n_nan > 0:
                    thres = np.max([block_data.shape[0] * .05, 3])
                    accept = False if n_nan > thres else True
                    if accept:
                        print(f'sub-{sub_id}, block-{int(block_id)} has {n_nan} nan values, accept: {accept}')
                        # fill the nan row with a random action from 
                        # get action space
                        nA = 3#int(block_data['block_type'].unique()[0])
                        block_data.loc[nan_idx, 'a'] = np.random.choice(nA, size=n_nan)
                        # update the reward 
                        block_data.loc[nan_idx, 'r'] = (
                            block_data.loc[nan_idx, 'a'] == 
                            block_data.loc[nan_idx, 'cor_a']
                        ).astype(int)
                    else:
                        print(f'sub-{sub_id}, block-{int(block_id)} has {n_nan} nan values, accept: {accept}')

                if accept:
                    sub_data_for_fit[block_id] = block_data[coi]
                    for_analysis.append(block_data[coi])

            # save the data for fit
            data_for_fit[sub_id] = sub_data_for_fit

            # ----------------- preprocess the control data ----------------- #
            sub_data_memory = {}
            block_lst = mem_data['block_id'].unique()
            for block_id in block_lst:
                block_data = mem_data.query(f'block_id=={block_id}').reset_index(drop=True)
                block_data['sub_id'] = f'sub-{sub_id}'
                block_data = block_data
                memory_data.append(block_data)            

    # save the for analysis data
    for_analysis = pd.concat(for_analysis)
    for_analysis.to_csv(f'{pth}/data/{data_set}-human.csv')
    # save the memory data 
    memory_data = pd.concat(memory_data)
    memory_data.to_csv(f'{pth}/data/{data_set}-mem-human.csv')
    # save the preprocessed data
    with open(f'{pth}/data/{data_set}.pkl', 'wb')as handle:
        pickle.dump(data_for_fit, handle)

if __name__ == '__main__':
    
    # set the random seed, used for nan-value filling
    np.random.seed(420)

    # preprocess collins 12 data 
    if args.data_set == 'setsize-collins12':
        pre_process_12()

    # preprocess collins 14 data 
    if args.data_set == 'setsize-collins14':
        pre_process_14()

    if args.data_set == 'setsize-uncomp':
        pre_process_uncomp('setsize-uncomp')

    if args.data_set == 'setsize-cluster':
        pre_process_cluster('setsize-cluster')