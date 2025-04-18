import argparse 
import os 
import pickle
import datetime 
import numpy as np
import pandas as pd
import subprocess
from copy import deepcopy


from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',     '-d', help='choose data set', default='setsize-collins12')
parser.add_argument('--agent_name',   '-n', help='data-generting agent', default='ecRL')
parser.add_argument('--method',       '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--algorithm',    '-a', help='fitting algorithm', type = str, default='BFGS')
# parser.add_argument('--other_agents', '-o', help='fitted agent', nargs='+', required=True)
parser.add_argument('--n_cores',      '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',         '-s', help='random seed', type=int, default=420)
parser.add_argument('--type',         '-t', help='type of recovery', type=str, default='param')
args = parser.parse_args()
args.agent = eval(args.agent_name)
args.env_fn = eval(f'{args.data_set.split("-")[0]}_task')
args.group = 'group' if args.method=='hier' else 'ind'
pth = os.path.dirname(os.path.abspath(__file__))


# -------------------------------- #
#        PARAMETER RECOVERY        #
# -------------------------------- #

def syn_data_param_recover_paral(pool, data, args, n_samp=20):

    # set seed, conditions
    rng = np.random.RandomState(args.seed+1)
    n_param = args.agent.n_params

    # get median parameters 
    fname = f'{pth}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'
    with open(fname, 'rb') as handle: fit_info = pickle.load(handle)
    param = np.median(np.vstack([item['param'] for item in fit_info.values()]), axis=0)

    # create sythesize params for different conditions
    truth_params = {p: [] for p in args.agent.p_names}
    truth_params['sub_id'] = []

    # the index paramters of interest
    poi_id = [args.agent.p_names.index(p) for p in args.agent.p_poi]

    # get params for parameter recovery 
    bnds = args.agent.p_pbnds
    n_sub = len(poi_id)*n_samp
    for sub_id in range(n_sub):
        p_temp = param.copy()
        for i in poi_id:
            p_temp[i] = bnds[i][0]+rng.rand()*(bnds[i][1]-bnds[i][0])
        for i in range(args.agent.n_params):
            truth_params[args.agent.p_names[i]].append(p_temp[i])
        truth_params['sub_id'].append(sub_id)
        
    ## save the ground turth parameters             
    truth_params_lst = pd.DataFrame.from_dict(truth_params)
    fname = f'{pth}/data/{args.data_set}-param_recover-{args.agent_name}.csv'
    truth_params_lst.to_csv(fname)
    
    ## start simulate with the generated parameters  
    res = [pool.apply_async(syn_data_param_recover, args=[row, data, args.seed+2+2*i])
                            for i, row in truth_params_lst.iterrows()]
    data_for_recovery = {}
    sub_lst = truth_params_lst['sub_id']
    for i, p in enumerate(res):
        data_for_recovery[sub_lst[i]] = p.get() 
    
    fname = f'{pth}/data/{args.data_set}-param_recover-{args.agent_name}.pkl'
    with open(fname, 'wb')as handle:
        pickle.dump(data_for_recovery, handle)
    print(f'Synthesize data (param recovery) for {args.agent_name} has been saved!')

def syn_data_param_recover(row, data, seed):

    # create random state 
    rng = np.random.RandomState(seed)
    model = wrapper(args.agent, env_fn=args.env_fn)
    # sample a subject task 
    sub_lst = list(data.keys())
    sub_id = rng.choice(sub_lst)
    task_data = data[sub_id]
    # get the target parameters 
    param = list(row[args.agent.p_names].values)
    # synthesize the data 
    recovery_data = model.sim(task_data, param, rng)
    recovery_data = recovery_data.drop(columns=args.agent.voi)
    # split the block according to the block id 
    sub_data = {}
    for block_id in recovery_data['block_id'].unique():
        sub_data[block_id] = recovery_data.query(f'block_id=={block_id}').reset_index(drop=True)
    return sub_data

def param_recover(args):

    ## STEP 0: GET PARALLEL POOL
    print(f'Parameter recovering {args.agent_name}...')
    pool = get_pool(args)

    ## STEP 1: SYTHESIZE FAKE DATA FOR PARAM RECOVER
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
    syn_data_param_recover_paral(pool, data, args, n_samp=20)
    pool.close() 

    ## STEP 2: REFIT THE MODEL TO THE SYTHESIZE DATA 
    cmand = ['python', 'm1_fit.py', f'-d={args.data_set}-param_recover-{args.agent_name}',
              f'-n={args.agent_name}', '-s=420', '-f=40',
              '-c=40', f'-m={args.method}', f'-a={args.algorithm}']
    subprocess.run(cmand)


# -------------------------------- #
#          MODEL RECOVERY          #
# -------------------------------- #

def model_recover(args, n_sub=40, n_samp=10):

    ## STEP 0: GET PARALLEL POOL
    pool = get_pool(args)

    ## STEP 1: SYTHESIZE FAKE DATA FOR PARAM RECOVER
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
    syn_data_model_recover_paral(pool, data, args, n_sub=n_sub, n_samp=n_samp)
    pool.close() 

    ## STEP 2: REFIT THE OTHER MODEL TO THE SYTHESIZE DATA 
    for agent_name in args.other_agents:
        cmand = ['python', 'm2_fit.py', f'-d={args.data_set}-{args.agent_name}',
                f'-n={agent_name}', '-s=420', '-f=50', '-c=50', 
                f'-m={args.method}', f'-a={args.algorithm}']
        subprocess.run(cmand)

def syn_data_model_recover_paral(pool, data, args, n_sub=30, n_samp=10):

    # get parameters 
    fname  = f'{pth}/fits/{args.data_set}/fit_sub_info'
    fname += f'-{args.agent_name}-{args.method}.pkl'      
    with open(fname, 'rb')as handle: fit_info_orig = pickle.load(handle)

    ## create a sub list of subject list 
    sub_lst_orig = list(fit_info_orig.keys())
    if 'group' in sub_lst_orig: sub_lst_orig.pop(sub_lst_orig.index('group'))
    # select subject for recovery 
    rng = np.random.RandomState(args.seed)
    # Use random choice without replacement to select subjects
    sub_lst = rng.choice(sub_lst_orig, size=n_sub, replace=False)
    fit_param = {k: fit_info_orig[k]['param'] for k in sub_lst}

    # create the synthesize data for the chosen sub
    res = [pool.apply_async(syn_data_model_recover, 
                    args=(data, fit_param[sub_id], sub_id, args.seed*i, n_samp))
                    for i, sub_id in enumerate(sub_lst)]

    syn_data = {}
    for _, p in enumerate(res):
        sim_data_all = p.get() 
        for sub_id in sim_data_all.keys():
            syn_data[sub_id] = sim_data_all[sub_id]

    # save for fit 
    with open(f'{pth}/data/{args.data_set}-{args.agent_name}.pkl', 'wb')as handle:
        pickle.dump(syn_data, handle)
    print(f'  {n_sub} Syn data for {args.agent_name} has been saved!')

def syn_data_model_recover(task_data, param, sub_id, seed, n_samp=10):

    # create random state 
    rng = np.random.RandomState(seed)
    model = wrapper(args.agent, args.env_fn)

    # synthesize the data and save
    
    task_lst = rng.choice(list(task_data.keys()), size=n_samp, replace=False)
    
    sim_data_all = {}
    for i, task_id in enumerate(task_lst):
        sample_id = f'{sub_id}-{i}'
        sim_data = {} 
        block_ind = task_data[task_id]
        for block_id in block_ind:
            task = task_data[task_id][block_id]
            sim_sample = model.sim({i: task}, param, rng=rng)
            sim_sample = sim_sample.drop(columns=model.agent.voi)
            sim_sample['sample_id'] = sample_id
            sim_data[block_id] = sim_sample  
        sim_data_all[sample_id] = deepcopy(sim_data)

    return sim_data_all

if __name__ == '__main__':

    eval(f'{args.type}_recover')(args)