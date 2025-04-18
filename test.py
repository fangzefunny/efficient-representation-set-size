import os 
import numpy as np 
import pandas as pd
from scipy.special import softmax 
import pingouin as pg 

import matplotlib.pyplot as plt 
import seaborn as sns 

#from utils.env import frozen_lake
import sys
sys.path.append('..')
from utils.viz import viz
from utils.model import *
from utils.env_fn import *
from utils.fig_fn import *
viz.get_style()


def sim(data_set, agent_name, params=None, sub_lst=None, scale=1):
    agent = eval(agent_name)
    model = wrapper(agent, setsize_task)
    with open(f'data/{data_set}.pkl', 'rb') as f:
        data = pickle.load(f)
    if sub_lst is None: sub_lst = data.keys()
    with open(f'fits/{data_set}/fit_sub_info-{agent_name}-mle.pkl', 'rb') as f:
        fit_info = pickle.load(f)
    rng = np.random.RandomState(2025)
    sim_data = []
    for sub_id in sub_lst:
        sub_data = data[sub_id]
        if params is not None: 
            params_to_fit = agent.link_params(params)
        else:
            params_to_fit = fit_info[sub_id]['param']*scale
        sim_datum = model.sim(sub_data, params_to_fit, rng)
        sim_data.append(sim_datum)
    sim_data = pd.concat(sim_data)
    return sim_data


tar_datum = sim('setsize-collins12', 'ecRL')


