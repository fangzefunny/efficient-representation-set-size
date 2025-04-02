import argparse 
import os 
import pickle
import datetime 
import numpy as np
import pandas as pd

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--n_fit',      '-f', help='fit times', type=int, default=1)
parser.add_argument('--data_set',   '-d', help='which_data', type=str, default='setsize-collins14')
parser.add_argument('--method',     '-m', help='methods, mle or map', type=str, default='mle')
parser.add_argument('--algorithm',  '-a', help='fitting algorithm', type=str, default='BFGS')
parser.add_argument('--agent_name', '-n', help='choose agent', default='RLWM')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=420)
parser.add_argument('--cv',         '-v', help='if do cross validation', type=bool, default=False)
args = parser.parse_args()
args.agent = eval(args.agent_name)
args.group = 'group' if args.method=='hier' else 'ind'
args.env = eval(f'{args.data_set.split("-")[0]}_task')
if args.cv: args.data_set = f'{args.data_set}-cv'

# find the current path, create the folders if not existed
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/fits', f'{pth}/fits/{args.data_set}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

def fit(pool, data, args):
    '''Find the optimal free parameter for each model 
    '''
    ## declare environment 
    model = wrapper(args.agent, args.env)

    ## fit list
    fname = f'{pth}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'
    if os.path.exists(fname):
        # load the previous fit resutls
        with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
        fitted_sub_lst = [k for k in fit_sub_info.keys()]
    else:
        fitted_sub_lst = []
        fit_sub_info = {}

    ## Start 
    start_time = datetime.datetime.now()
    sub_start  = start_time

    ## Fit params to each individual 
    if args.group == 'ind':
        done_subj = len(fitted_sub_lst)
        all_subj  = len(data.keys()) 
        for sub_idx in data.keys(): 
            if sub_idx not in fitted_sub_lst:  
                print(f'Fitting {args.agent_name} subj {sub_idx}, progress: {(done_subj*100)/all_subj:.2f}%')
                fit_info = model.fit(data[sub_idx], args.method, args.algorithm, 
                                     pool=pool, seed=args.seed, n_fits=args.n_fit,
                                     verbose=False, init=False)
                fit_sub_info[sub_idx] = fit_info
                with open(fname, 'wb')as handle: 
                    pickle.dump(fit_sub_info, handle)
                sub_end = datetime.datetime.now()
                print(f'\tLOSS:{-fit_info["log_post"]:.4f}, using {(sub_end - sub_start).total_seconds():.2f} seconds')
                sub_start = sub_end
                done_subj += 1
    elif args.group == 'group':
        fit_sub_info = fit_hier(pool, data, model, fname,  
                                seed=args.seed, n_fits=args.n_fit)
        with open(fname, 'wb')as handle: 
            pickle.dump(fit_sub_info, handle)
                
    ## END!!!
    end_time = datetime.datetime.now()
    print('\nparallel computing spend {:.2f} seconds'.format(
            (end_time - start_time).total_seconds()))
    
def fit_cv(pool, data, args):
    '''Find the optimal free parameter for each model 
    '''
    ## declare environment 
    model = wrapper(args.agent, args.env)

    ## fit list
    fname = f'{pth}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'
    if os.path.exists(fname):
        # load the previous fit resutls
        with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
        fitted_sub_lst = [k for k in fit_sub_info.keys()]
    else:
        fitted_sub_lst = []
        fit_sub_info = {}

    ## Start 
    start_time = datetime.datetime.now()
    sub_start  = start_time

    ## Fit params to each individual 
    done_subj = len(fitted_sub_lst)
    all_subj  = len(data.keys()) 
    for sub_id in data.keys(): 
        if sub_id not in fitted_sub_lst:
            train_data = data[sub_id]['train']['data']
            train_weights = data[sub_id]['train']['weights']
            test_data = data[sub_id]['test']
            print(f'Fitting {args.agent_name} subj {sub_id}, progress: {(done_subj*100)/all_subj:.2f}%')
            fit_info = model.fit(train_data, args.method, args.algorithm, 
                                    pool=pool, seed=args.seed, n_fits=args.n_fit,
                                    init=False, train_weights=train_weights, 
                                    test_data=test_data, verbose=False)
            fit_sub_info[sub_id] = fit_info
            with open(fname, 'wb')as handle: 
                pickle.dump(fit_sub_info, handle)
            sub_end = datetime.datetime.now()
            print(f'\tLOSS:{-fit_info["log_post"]:.4f}, using {(sub_end - sub_start).total_seconds():.2f} seconds')
            sub_start = sub_end
            done_subj += 1

    ## END!!!
    end_time = datetime.datetime.now()
    print('\nparallel computing spend {:.2f} seconds'.format(
            (end_time - start_time).total_seconds()))

if __name__ == '__main__':

    ## STEP 0: GET PARALLEL POOL
    pool = get_pool(args)

    ## STEP 1: LOAD DATA 
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
  
    ## STEP 2: FIT
    if args.cv:
        fit_cv(pool, data, args)
    else:
        fit(pool, data, args)   
    pool.close()