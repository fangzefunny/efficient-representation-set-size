import pickle
import os 
import numpy as np 
import pandas as pd 
import pingouin as pg 
import matplotlib.pyplot as plt
import seaborn as sns 
from .model import *
from .viz import viz 
from .stats import *

pth = os.path.dirname(os.path.abspath(__file__))
    
# ----------------------------------------------------------------------------- #
#                             Behavioral Patterns                               # 
# ----------------------------------------------------------------------------- #

def get_asymptote_acc(data):
    asymptote = {'block_type': [], 'acc': [], 'sub_id': []}
    for sub_id in data['sub_id'].unique():
        sub_data = data.query(f'sub_id=="{sub_id}"').reset_index(drop=True)
        block_lst = sub_data['block_id'].unique()
        for block_id in block_lst:
            block_data = sub_data.query(f'block_id=={block_id}').reset_index(drop=True)
            max_rep = block_data['repetitions'].max().astype(int)
            reps = list(range(max_rep-1, max_rep+1))
            sel_data = block_data.query(f'repetitions in {reps}')
            set_size = sel_data['block_type'].unique()[0].astype(int)
            acc = sel_data['r'].mean()
            asymptote['sub_id'].append(sub_id)
            asymptote['block_type'].append(set_size)
            asymptote['acc'].append(acc)
    asymptote = pd.DataFrame(asymptote)
    sel_data = asymptote.groupby(by=['block_type', 'sub_id']
            ).agg({'acc': 'mean'}).reset_index()
    return sel_data

def two_behavioral_patterns(data, axs, mode='human', palette=viz.sns_purple,
                            optimal_curve=True):
    data = data.copy()
    if mode!='human': data['r'] = data['acc']
    sel_data = get_asymptote_acc(data)    # the set size effect
    ax = axs[0]
    sns.lineplot(x='repetitions', y='r', 
                    data=data.query(f'repetitions<10'), 
                    errorbar='se', #err_style='bars', err_kws={'capsize': 3},
                    hue='block_type', lw=2, 
                    palette=palette, legend=False,
                    ax=ax)
    if optimal_curve:
        # add the optimal curve
        opt_curve = [1/3, 2/3] + [1]*7
        ax.plot(range(1, 10), opt_curve, color='k', lw=1.5, ls='--')
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_box_aspect(.85)
    # the asymptote accuracy
    ax = axs[1]
    viz.violin(ax, data=sel_data, y='acc',
                x='block_type', order=[2, 3, 4, 5, 6],
                errorbar='se', scatter_size=2.2,
                scatter_type='strip',
                scatter_alpha=.7,
                err_capsize=.17, 
                palette=palette)
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_ylim(.75, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Set size')
    ax.set_box_aspect(.85)

def two_behavioral_patterns_with_target(data, tar_data_set, axs, mode='human'):
    data = data.copy()
    if mode!='human': data['r'] = data['acc']
    human_data = get_beh_data(tar_data_set, 'human')
    sel_data = get_asymptote_acc(data)    # the set size effect
    ax = axs[0]
    sns.lineplot(x='repetitions', y='r', 
                    data=data.query(f'repetitions<10'), 
                    errorbar=('ci', 95), #err_style='bars', err_kws={'capsize': 3},
                    hue='block_type', lw=0, 
                    ax=ax)
    # add human data as target
    sns.lineplot(x='repetitions', y='r', 
                    data=human_data.query(f'repetitions<10'), 
                    errorbar=('ci', 95), err_style='bars', err_kws={'capsize': 3},
                    hue='block_type', lw=1, 
                    ax=ax)

    # add the optimal curve
    opt_curve = [1/3, 2/3] + [1]*7
    ax.plot(range(1, 10), opt_curve, color=viz.Blue, lw=3, ls='--')
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_box_aspect(.85)
    # the asymptote accuracy
    ax = axs[1]
    viz.violin(ax, data=sel_data, y='acc',
                x='block_type', order=[2, 3, 4, 5, 6],
                errorbar=('ci', 95), scatter_size=3.3,
                err_capsize=.17, 
                palette=viz.snsPalette)
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_ylim(.75, 1.05)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Set size')
    ax.set_box_aspect(.85)

# ----------------------------------------------------------------------------- #
#                              General functions                                # 
# ----------------------------------------------------------------------------- #

def get_beh_data(data_set, model, method='mle', cross_valid=True):
    if model=='human':
        fname = f'{pth}/../data/{data_set}-human.csv'
    else:
        cv = '-cv' if cross_valid else ''
        fname = f'{pth}/../simulations/{data_set}{cv}/{model}/sim-{method}.csv'
    return pd.read_csv(fname)

def get_fit_param(data_set, model, method='mle', poi=None, p_trans=False):
    fname = f'{pth}/../fits/{data_set}/fit_sub_info-{model}-{method}.pkl'
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    if poi is None: poi = eval(model).p_names
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
    params = {p: [] for p in poi}
    params['sub_id'] = []
    for sub_id in sub_lst:
        params['sub_id'].append(sub_id)
        for p, fn in zip(poi, eval(model).p_trans):
            idx = fit_sub_info[sub_id]['param_name'].index(p) 
            pvalue = fit_sub_info[sub_id]['param'][idx].copy()
            params[p].append(fn(pvalue))
    return pd.DataFrame.from_dict(params)

def get_llh_score(data_set, models, method, 
                  use_bic=False,
                  relative=True):
    '''Get likelihood socres

    Inputs:
        models: a list of models for evaluation
    
    Outputs:
        crs: nll, aic and bic score per model per particiant
        pxp: pxp score per model per particiant
    '''
    tar = models[0] 
    fit_sub_info = []
    for i, m in enumerate(models):
        with open(f'{pth}/../fits/{data_set}/fit_sub_info-{m}-{method}.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        log_post = [fit_info[idx]['log_post'] for idx in subj_lst]
        bic      = [fit_info[idx]['bic'] for idx in subj_lst]
        h        = [fit_info[idx]['H'] for idx in subj_lst] if use_bic==False else 0
        n_param  = fit_info[list(subj_lst)[0]]['n_param']
        fit_sub_info.append({
            'log_post': log_post, 
            'bic': bic, 
            'n_param': n_param, 
            'H': h,
        })
    # get bms 
    bms_results = fit_bms(fit_sub_info, use_bic=use_bic)

    ## combine into a dataframe 
    cols = ['NLL', 'AIC', 'BIC', 'model', 'sub_id']
    crs = {k: [] for k in cols}
    for m in models:
        with open(f'{pth}/../fits/{data_set}/fit_sub_info-{m}-{method}.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        nll = [-fit_info[idx]['log_like'] for idx in subj_lst]
        aic = [fit_info[idx]['aic'] for idx in subj_lst]
        bic = [fit_info[idx]['bic'] for idx in subj_lst]
        crs['NLL'] += nll
        crs['AIC'] += aic
        crs['BIC'] += bic
        crs['model'] += [m]*len(nll)
        crs['sub_id'] += list(subj_lst)
    crs = pd.DataFrame.from_dict(crs)
    for c in ['NLL', 'BIC', 'AIC']:
        tar_crs = len(models)*list(crs.query(f'model=="{tar}"')[c].values)
        subtrack = tar_crs if relative else 0
        crs[c] -= subtrack
    pxp = pd.DataFrame.from_dict({'pxp': bms_results['pxp'], 'model': models})
    return crs, pxp 

def model_compare_group(axs, llh_table, pxp, models, cr='bic'):
    ax = axs[0]
    palette = [eval(m).color for m in models]
    viz.violin(data=llh_table, x=cr.upper(), y='model', 
            order=models, orient='h', palette=palette,
            mean_marker_size=4.5, scatter_size=2, scatter_alpha=.75,
            errorbar='sd', errorcolor=[.2]*3, err_capsize=.16,
            errorlw=2,
            ax=ax)
    ax.axvline(x=0, ymin=0, ymax=1, color='k', ls='--', lw=1.5)
    lbl = [eval(m).name for m in models]
    ax.set_yticks(list(range(len(models)))) 
    ax.set_yticklabels(lbl)
    ax.spines['left'].set_position(('axes',-0.1))
    #for pos in ['bottom','left']: ax.spines[pos].set_linewidth(2)
    ax.set_ylabel('')
    ax.set_xlabel(r'$\Delta$'+f'{cr.upper()}')
    ax.set_box_aspect(1.8)

    ax = axs[1]
    palette = [eval(m).color for m in models]
    sns.barplot(ax=ax, data=pxp, y='model', x='pxp', 
                hue='model', edgecolor=[.2]*3, lw=1.75,
                palette=palette)
    #for pos in ['bottom','left']: ax.spines[pos].set_linewidth(2)
    ax.set_xlabel('PXP')
    ax.set_box_aspect(1.8)

def model_compare_ind(ax, data_set, models, n_data=None, method='mle', cr='bic'):
    crs_table = [] 
    for m in models:
        fname = f'../fits/{data_set}/fit_sub_info-{m}-{method}.pkl'
        with open(fname, 'rb')as handle: fit_info = pickle.load(handle)
        sub_lst = list(fit_info.keys())
        if 'group' in sub_lst: sub_lst.pop(sub_lst.index('group'))
        crs = {'sub_id': [], 'aic': [], 'bic': [], 'model': []}
        for sub_id in sub_lst:
            crs['sub_id'].append(sub_id)
            crs['aic'].append(fit_info[sub_id]['aic'])
            crs['bic'].append(fit_info[sub_id]['bic'])
            crs['model'].append(m)
        crs_table.append(pd.DataFrame.from_dict(crs))
    crs_table = pd.concat(crs_table, axis=0, ignore_index=True)
    sel_table = crs_table.pivot_table(
        values=cr,
        index='sub_id',
        columns='model',
        aggfunc=np.mean,
    ).reset_index()
    sel_table[f'min_{cr}'] = sel_table.apply(
        lambda x: np.min([x[m] for m in models]), 
        axis=1)
    sort_table = sel_table.sort_values(by=f'min_{cr}').reset_index()
    sort_table['sub_seq'] = sort_table.index

    for m in models:
        model_fn = eval(m)
        sns.scatterplot(x='sub_seq', y=m, data=sort_table,
                        marker=model_fn.marker,
                        size=model_fn.size,
                        color=model_fn.color,
                        alpha=model_fn.alpha,
                        linewidth=1.1,
                        edgecolor=model_fn.color if model_fn.marker in ['s', 'o', '^'] else 'none',
                        facecolor='none'if model_fn.marker in ['s', 'o'] else model_fn.color,
                        label=model_fn.name,
                        ax=ax)

    #n_data = (12*8+6*8)*2
    if n_data is not None:
        ax.axhline(y=-np.log(1/2)*n_data*2, xmin=0, xmax=1,
                        color='k', lw=1, ls='--')
    ax.set_xlim([-5, sort_table.shape[0]+5])
    ax.legend(loc='upper left')
    ax.spines['left'].set_position(('axes',-0.02))
    ax.set_xlabel(f'Participant index\n(sorted by the minimum {cr} score over all models)')
    ax.set_ylabel(cr.upper())

def sim(agent_fn, params, block_type='block', seed=1234, stop=1000):

    block_data = forget_task(block_type).instan(seed*2+1)
    folder = 'block0'
    env = forget_task(block_type)
    env.set_label(folder)
    model = agent_fn(env, params=params)

    ## init a blank dataframe to store variable of interest
    rng = np.random.RandomState(seed)
    col = forget_task.voi
    init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
    pred_data = pd.DataFrame(init_mat, columns=col)  

    ## loop to simulate the responses in the block
    for t, row in block_data.iterrows():

        #print(t)

        # simulate the data 
        subj_voi = env.sim_fn(row, model, rng)
        
        # record the stimulated data
        for i, v in enumerate(env.voi): 
            pred_data.loc[t, v] = subj_voi[i]
        
        if t>=stop: break

    # drop nan columns
    pred_data = pred_data.dropna(axis=1, how='all')
    
    return pd.concat([block_data, pred_data], axis=1), model

def n_sim(agent_fn, params, block_type=['block', 'interleave'], seed=1234, n=20):
    data = []
    i = 0
    for _ in range(n):
        for block_type in ['block', 'interleave']:
            datum, model = sim(agent_fn, params, block_type, seed+i)
            datum['sub_id'] = f'sim_{i}'
            data.append(datum)
            i += 1
    return pd.concat(data, axis=0, ignore_index=True), model

def get_param(data_set:str, model:str, param_names:list):
    '''Create a param table
    '''
    fname = f'{pth}/../fits/{data_set}/fit_sub_info-{model}-mle.pkl'
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    sub_lst = list(fit_sub_info.keys())
    if 'group' in sub_lst: sub_lst.pop(sub_lst.index(sub_lst))
    param_table = {p: [] for p in param_names}
    param_table['sub_id'] = []
    fns = eval(model).p_trans
    for sub_id in sub_lst:
        item = fit_sub_info[sub_id]
        param_table['sub_id'].append(sub_id)
        for p_name in param_names:
            idx = item['param_name'].index(p_name)
            param = fns[idx](item['param'][idx])
            param_table[p_name].append(param)

    return pd.DataFrame.from_dict(param_table)
     
