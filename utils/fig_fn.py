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

def learning_curve(ax, data, palette=[viz.Blue, viz.Red]):
    # some preprocessing
    data['s'] = data['s'].apply(lambda x: int(x))
    # show the curve 
    sel_data = data.query('stage=="train"').groupby(
    by=['sub_id', 'trial', 'block_type', 'subtask'])['r'].mean().reset_index()
    # visualize
    
    query_block_subtask_A = 'subtask=="A"&block_type=="block"'
    trial_a_end = sel_data.query(query_block_subtask_A)['trial'].max()
    query_block_subtask_B = 'subtask=="B"&block_type=="block"'
    trial_b_end = sel_data.query(query_block_subtask_B)['trial'].max()

    sns.lineplot(x='trial', y='r', data=sel_data.query(query_block_subtask_A), 
                    err_style='band',
                    color=palette[0],
                    lw=3,
                    err_kws={'alpha':.2},
                    ax=ax, 
                    label='blocked',
                    solid_capstyle='butt')
    sns.lineplot(x='trial', y='r', data=sel_data.query(query_block_subtask_B), 
                    err_style='band',
                    color=palette[0],
                    lw=3,
                    err_kws={'alpha':.2},
                    ax=ax, 
                    label='blocked',
                    solid_capstyle='butt')
    sns.lineplot(x='trial', y='r', data=sel_data.query('block_type=="interleave"'), 
                    err_style='band',
                    color=palette[1], 
                    lw=3,
                    err_kws={'alpha':.2},
                    ax=ax, 
                    label='interleaved',
                    solid_capstyle='butt')
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=.5)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend().remove()
    ax.spines['left'].set_position(('axes',-0.05))
    ax.set_box_aspect(.65)
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    tick_label = [0, trial_a_end, trial_b_end]
    ax.set_xticks(tick_label)
    ax.set_xticklabels([int(x+1) for x in tick_label])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Trials')
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1.])
    
def learning_speed(ax, data, palette=[viz.Blue, viz.Red]):
    # some preprocessing
    data['s'] = data['s'].apply(lambda x: int(x))

    # show the curve 
    sel_data = data.query('stage=="train"').groupby(
        by=['sub_id', 'tps', 'block_type'])['r'].mean().reset_index()
    # visualize
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="block"'), 
                err_style='band',
                color=palette[0],
                err_kws={'alpha':.3},
                lw=3.5,
                ax=ax, 
                label='blocked',
                solid_capstyle='butt')
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="interleave"'), 
                err_style='band',
                color=palette[1],
                err_kws={'alpha':.3},
                lw=3.5,
                ax=ax, 
                label='interleaved',
                solid_capstyle='butt')
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=.5)
    ax.spines['left'].set_position(('axes',-0.08))
    ax.set_box_aspect(1.2)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1.])
  
def catastrophic_forgetting(ax, data, if_t_test=False):
    sel_data = data.query('stage=="test"').groupby(
        by=['sub_id', 'subtask', 'block_type'])['r'].mean().reset_index()
    # visualize
    viz.violin(ax, data=sel_data, y='r',
            x='subtask', order=['A', 'B'],
            hue='block_type', hue_order=['block', 'interleave'],
            palette=[viz.Blue, viz.Red],
            scatter_size=4, scatter_alpha=.85,
            mean_marker_size=4.5, errorbar='se', 
            scatter_edge_color='w', scatter_lw=.15, errorlw=2,
            errorcolor=[.2,.2,.2],
            err_capsize=.16)
    if if_t_test:
        print('\tTest performance: ')
        x = sel_data.query('block_type=="block"&subtask=="A"')['r']
        y = sel_data.query('block_type=="block"&subtask=="B"')['r']
        t_test(x, y, paired=True, title='\tBLOCK A-B')
    ax.axhline(y=.5, xmin=0, xmax=1, lw=.5, 
            color=[.2, .2, .2], ls='--')
    ax.set_xticks([0, 1])
    ax.set_ylim([0, 1])
    ax.set_box_aspect(.8)
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.set_xticklabels(['A', 'B'])
    ax.spines['left'].set_position(('axes',-0.08))
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1.])

def forward_interference(ax, data):
    data['s'] = data['s'].apply(lambda x: int(x))
    get_subtask = lambda x: 'A' if x<4 else 'B'
    data['subtask'] = data['s'].apply(get_subtask)
    # show the curve 
    sel_data = data.query('stage=="train"').groupby(
    by=['sub_id', 'tps', 'block_type', 'subtask'])['r'].mean().reset_index()
    # visualize
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="block"'), 
                 errorbar='se',
                 style='subtask',
                 color=viz.Blue,
                 lw=2.75,
                 ax=ax)
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=.5)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines['left'].set_position(('axes',-0.08))
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.set_box_aspect(1.2)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Trials')
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1.])

def get_test_per_s(ax, data, block_type, s_space=False):
    # some preprocessing
    sel_data = data.query(f'stage=="test"&block_type=="{block_type}"').groupby(
    by=['sub_id', 's'])['r'].mean().reset_index()
    sel_data['agent'] = 'model'
    color = viz.Blue if block_type=='block' else viz.Red
    viz.violin(ax=ax, data=sel_data, y='r', 
             x='s', order=s_space if s_space else np.arange(8),
            palette=[color]*8, errorbar='se', 
            errorcolor=[.2]*3, err_capsize=.17, mean_marker_size=4.5,
            errorlw=2, scatter_size=3, scatter_lw=.1)
    ax.axhline(y=.5, xmin=0, xmax=1, lw=1, 
            color=[.2, .2, .2], ls='--')
    ax.axvline(x=3.5, ymin=0, ymax=1, lw=1, 
            color=[.2, .2, .2], ls='--')
    ax.set_ylim([0, 1])
    ax.set_box_aspect(.65)
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.spines['left'].set_position(('axes',-0.04))
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([-.15, 1.15])

def get_learn_gap(data):
    data['s'] = data['s'].apply(lambda x: int(x)) 
    # learning speed in task B
    test_A = data.query('block_type=="block"&stage=="test"&subtask=="A"'
                    ).groupby('sub_id')['r'].mean().add(-.5
                    ).reset_index().rename(columns={'r': 'test_A'})
    m = test_A['test_A'].median()
    valid_sub_lst = test_A.query(f'test_A>={m}')['sub_id'].to_list()
    # test A
    learn_block_B = data.query('block_type=="block"&stage=="train"&subtask=="B"'
                    ).groupby('sub_id')['r'].mean().add(-.5
                    ).reset_index()
    learn_block_B['block_type'] = 'block B'
    learn_inter = data.query('block_type=="interleave"&stage=="train"'
                    ).groupby('sub_id')['r'].mean().add(-.5
                    ).reset_index()
    learn_inter['block_type'] = 'interleave'
    # combine 
    comb_data = pd.concat([learn_block_B, learn_inter], axis=0
                        ).query(f'sub_id in {valid_sub_lst}')
    block_data = comb_data.query('block_type=="block B"').rename(
            columns={'r': 'block_B_r'}).drop(columns='block_type')
    interleave_data = comb_data.query('block_type=="interleave"').rename(
            columns={'r': 'interleave_r'}).drop(columns='block_type')
    comb_data = block_data.merge(interleave_data, on='sub_id')
    comb_data['gap'] = comb_data['block_B_r'] - comb_data['interleave_r']
    return comb_data

def learn_gap_top50(ax, data):
    valid_sub_lst = get_learn_gap(data)['sub_id'].to_list()
    sel_data = data.query(f'sub_id in {valid_sub_lst}')
    q = 'stage=="train"&block_type=="block"&subtask=="B"'
    sns.lineplot(data=sel_data.query(q).groupby(
                    by=['sub_id', 'tps', 'block_type']
                    )['r'].mean().reset_index(), 
                x='tps', y='r',
                errorbar='se',
                err_style='band',
                color=viz.lBlue2,
                lw=3,
                ax=ax, 
                err_kws={'alpha':.6},
                label='blocked B',
                solid_capstyle='butt')
    q = 'stage=="train"&block_type=="interleave"'
    sns.lineplot(data=sel_data.query(q).groupby(
                    by=['sub_id', 'tps', 'block_type']
                    )['r'].mean().reset_index(),
                x='tps', y='r',
                errorbar='se',
                err_style='band',
                color=viz.lRed2,
                lw=3,
                ax=ax, 
                err_kws={'alpha':.6},
                label='interleaved',
                solid_capstyle='butt')
    ax.spines['left'].set_position(('axes',-0.08))
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=.5)
    ax.set_box_aspect(1.2)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1.])

def t_test_top50(ax, data, model_name, color, if_t_test=False):
    # some preprocessing
    gap_data = get_learn_gap(data)
    gap_data['model']=model_name
    if if_t_test:
        print('\tAccuracy Gap: ')
        t_test(gap_data['gap'], 0, title='human-gap:'.rjust(6))
    viz.violin(ax, data=gap_data, y='gap',
               x='model', order=[model_name],
               palette=[color],
               errorbar='se', errorlw=2, err_capsize=.17,
               scatter_edge_color='w', scatter_lw=.3,
               mean_marker_size=4.5, scatter_size=3)
    ax.axhline(y=0, xmin=0, xmax=1, color='k', lw=1, ls='--')
    ax.spines['left'].set_position(('axes',-0.08))
    #for pos in ['bottom','left']: ax.spines[pos].set_linewidth(2)
    ax.set_xlabel('')
    ax.set_ylim([-.2, .55])
    ax.set_box_aspect(1.6)

def show_six_patterns(data, figsize=(4, 5)):
    '''Show all six behavioral patterns'''
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    axes = [fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
    learning_curve(axes[0], data)
    axes[0].set_box_aspect(.6)
    learning_speed(axes[1], data)
    axes[1].set_box_aspect(1.2)
    catastrophic_forgetting(axes[2], data, if_t_test=True)
    axes[2].set_box_aspect(.65)
    forward_interference(axes[3], data)
    axes[3].set_box_aspect(1.2)
    get_test_per_s(axes[4], data, block_type='block')
    axes[4].set_box_aspect(.6)
    learn_gap_top50(axes[5], data)
    axes[5].set_box_aspect(1.2)
    for i, ax in enumerate(axes): 
        if i%2==1: ax.set_ylabel('')
        ax.set_xlabel('')

# ----------------------------------------------------------------------------- #
#                               Human vs. Model                                 # 
# ----------------------------------------------------------------------------- #

def learning_speed_target(ax, data, with_target_data_set=False):
    # show the curve 
    sel_data = data.query('stage=="train"').groupby(
        by=['sub_id', 'tps', 'block_type'])['r'].mean().reset_index()
    # get target
    target_data = get_beh_data(with_target_data_set, 'human')
    target_data['subtask'] = target_data['s'].apply(lambda x: int(x>=4))
    target_data = target_data.query('stage=="train"').groupby(
            by=['sub_id', 'tps', 'block_type'])['r'].mean().reset_index()
    # visualize the model prediction 
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="block"'), 
                err_style='band',
                color=viz.Blue,
                lw=0,
                ax=ax, 
                err_kws={'alpha':.5}, 
                label='blocked')
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="interleave"'), 
                err_style='band',
                color=viz.Red, 
                lw=0,
                ax=ax, 
                err_kws={'alpha':.5}, 
                label='interleaved')
    # show the target data 
    sns.lineplot(x='tps', y='r', data=target_data.query('block_type=="block"'), 
                color=viz.Blue,
                lw=0,
                ax=ax, 
                err_style='bars', errorbar="se", 
                err_kws={'capsize': 2.5, 'elinewidth': 1.25, 'capthick': 1.45})
    sns.lineplot(x='tps', y='r', data=target_data.query('block_type=="interleave"'), 
                color=viz.Red, 
                lw=0,
                ax=ax, 
                err_style='bars', errorbar="se", 
                err_kws={'capsize': 2.5, 'elinewidth': 1.25, 'capthick': 1.45})
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=1)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines['left'].set_position(('axes',-0.08))
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.set_box_aspect(1.2)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_ylim([.2, 1.05])

def forward_interference_target(ax, data, tar_data_set=False):
    # show the curve 
    sel_data = data.query('stage=="train"').groupby(
        by=['sub_id', 'tps', 'block_type', 'subtask'])['r'].mean().reset_index()
     # visualize the model prediction 
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="block"&subtask=="A"'), 
                err_style='band',
                color=viz.Blue,
                lw=2,
                ax=ax, 
                err_kws={'alpha':.5}, 
                label='blocked')
    sns.lineplot(x='tps', y='r', data=sel_data.query('block_type=="block"&subtask=="B"'), 
                err_style='band',
                color=viz.Blue, 
                lw=2,
                ls='--',
                ax=ax, 
                err_kws={'alpha':.3}, 
                label='interleaved')
    # get target
    if tar_data_set:
        target_data = get_beh_data(tar_data_set, 'human')
        target_data = target_data.query('stage=="train"').groupby(
                by=['sub_id', 'tps', 'block_type', 'subtask'])['r'].mean().reset_index()
        # show the target data 
        sns.lineplot(x='tps', y='r', data=target_data.query('block_type=="block"&subtask=="A"'), 
                    color=viz.Blue,
                    lw=0,
                    ax=ax, 
                    err_style='bars', errorbar="se", 
                    err_kws={'capsize': 2.5, 'elinewidth': 1.5, 'capthick': 1.75})
        sns.lineplot(x='tps', y='r', data=target_data.query('block_type=="block"&subtask=="B"'), 
                    color=viz.lBlue2, 
                    lw=0,
                    ax=ax, 
                    err_style='bars', errorbar="se", 
                    err_kws={'capsize': 2.5, 'elinewidth': 1.5, 'capthick': 1.75})
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=1)
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.spines['left'].set_position(('axes',-0.08))
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.set_box_aspect(1.2)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_ylim([.2, 1.05])

def catastrophic_forgetting_target(ax, pred_data, tar_data_set, block_type='block', color=viz.Blue):
    # preprocess the model prediction
    pred_data['subtask'] = pred_data['s'].apply(lambda x: 'A' if x<4 else 'B')
    pred_data['s'] = pred_data['s'].apply(lambda x: int(x))
    pred_data = pred_data.query(f'stage=="test"&block_type=="{block_type}"').groupby(
        by=['sub_id', 'subtask'])['r'].mean().reset_index()
    pred_data['agent'] = 'model'
    # preprocess the target data
    tar_data  = get_beh_data(tar_data_set, 'human')
    tar_data['s'] = tar_data['s'].apply(lambda x: int(x))
    tar_data = tar_data.query(f'stage=="test"&block_type=="{block_type}"').groupby(
        by=['sub_id', 'subtask'])['r'].mean().reset_index()
    tar_data['agent'] = 'human'
    # combine data
    comb_data = pd.concat([pred_data, tar_data], axis=0, ignore_index=True)    
    # visualize
    viz.violin_with_tar(ax=ax, 
            data=comb_data, y='r', 
            x='subtask', order=['A', 'B'], 
            color=color, errorbar='se',
            errorcolor=[.3]*3, err_capsize=.14,
            errorlw=1.3, scatter_size=3,
            mean_marker_size=1.2,
            hue='agent', hue_order=['human', 'model'])
    for sub_task in ['A', 'B']:
        x = comb_data.query(f'subtask=="{sub_task}"&agent=="human"')['r']
        y = comb_data.query(f'subtask=="{sub_task}"&agent=="model"')['r']
        t_test(x, y, title=f'{block_type.rjust(10)}, "{sub_task}"')
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.axhline(y=.5, xmin=0, xmax=1, lw=1, 
            color=[.2, .2, .2], ls='--')
    ax.set_xticks([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticklabels(['A', 'B'])
    ax.spines['left'].set_position(('axes',-0.08))
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')
    ax.set_box_aspect(1.2)
    ax.set_ylim([.2, 1.05])

def get_test_per_s_target(ax, data, block_type, tar_data_set, stimuli=np.arange(8)):
    # some preprocessing
    sel_data = data.query(f'stage=="test"&block_type=="{block_type}"').groupby(
    by=['sub_id', 's'])['r'].mean().reset_index()
    sel_data['agent'] = 'model'
    # preprocess the target data
    tar_data  = get_beh_data(tar_data_set, 'human')
    tar_data['s'] = tar_data['s'].apply(lambda x: int(x))
    tar_data = tar_data.query(f'stage=="test"&block_type=="{block_type}"').groupby(
        by=['sub_id', 's'])['r'].mean().reset_index()
    tar_data['agent'] = 'human'
    # combine data
    comb_data = pd.concat([sel_data, tar_data], axis=0, ignore_index=True) 
    # visualize
    color = viz.Blue if block_type=='block' else viz.Red
    viz.violin_with_tar(ax=ax, data=comb_data, y='r', 
            x='s', order=stimuli,
            color=color, errorbar='se',
            errorcolor=[.3]*3, err_capsize=.14,
            errorlw=1.3, scatter_size=3,
            mean_marker_size=1.2,
            hue='agent', hue_order=['human', 'model'])
    ax.axhline(y=.5, xmin=0, xmax=1, lw=1, 
            color=[.2, .2, .2], ls='--')
    ax.axvline(x=3.5, ymin=0, ymax=1, lw=1, 
            color=[.2, .2, .2], ls='--')
    ax.set_ylim([0, 1])
    ax.set_box_aspect(.6)
    ax.spines['left'].set_position(('axes',-0.08))
    ax.set_xlabel('')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([-.05, 1.1])

def learn_gap_top50_with_target(ax, data, tar_data_set='forget-control'):
    valid_sub_lst = get_learn_gap(data)['sub_id'].to_list()
    sel_data = data.query(f'sub_id in {valid_sub_lst}')
    q = 'stage=="train"&block_type=="block"&subtask=="B"'
    sns.lineplot(data=sel_data.query(q).groupby(
                    by=['sub_id', 'tps', 'block_type']
                    )['r'].mean().reset_index(), 
                x='tps', y='r',
                err_style='band',
                errorbar='se',
                color=viz.lBlue2,
                lw=0,
                ax=ax, 
                err_kws={'alpha':.7},
                solid_capstyle='butt')
    q = 'stage=="train"&block_type=="interleave"'
    sns.lineplot(data=sel_data.query(q).groupby(
                    by=['sub_id', 'tps', 'block_type']
                    )['r'].mean().reset_index(), 
                x='tps', y='r',
                err_style='band',
                errorbar='se',
                color=viz.lRed2,
                lw=0,
                ax=ax, 
                err_kws={'alpha':.7},
                solid_capstyle='butt')
    human_data = get_beh_data(tar_data_set, 'human')
    valid_sub_lst = get_learn_gap(human_data)['sub_id'].to_list()
    human_data = human_data.query(f'sub_id in {valid_sub_lst}')
    q = 'stage=="train"&block_type=="block"&subtask=="B"'
    sns.lineplot(data=human_data.query(q), 
                x='tps', y='r',
                errorbar='se',
                err_style='bars',
                color=viz.lBlue2,
                lw=0,
                ax=ax, 
                zorder=-10,
                solid_capstyle='butt',
                err_kws={'capsize': 2.5, 'elinewidth': 1.25, 'capthick': 1.45})
    q = 'stage=="train"&block_type=="interleave"'
    sns.lineplot(data=human_data.query(q), 
                x='tps', y='r',
                errorbar='se',
                err_style='bars',
                color=viz.lRed2,
                lw=0,
                ax=ax, 
                zorder=-10,
                solid_capstyle='butt',
                err_kws={'capsize': 2.5, 'elinewidth': 1.25, 'capthick': 1.45})
    ax.spines['left'].set_position(('axes',-0.08))
    #for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(2)
    ax.axhline(y=.5, xmax=0, xmin=1, ls='--', color='k', lw=.5)
    #ax.legend().remove()
    ax.set_box_aspect(1.2)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Repetitions')
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1.])

def t_test_top50_target(ax, data, tar_data_set, color):
    # some preprocessing
    model_data = get_learn_gap(data)
    model_data['model'] = 'model' 
    t_test(model_data['gap'], 0, title='gap-model:'.rjust(6))
    human_data = get_learn_gap(get_beh_data(tar_data_set, 'human'))
    human_data['model'] = 'human'
    t_test(human_data['gap'], 0, title='gap-human'.rjust(6))
    # combine the data 
    comb_data = pd.concat([model_data, human_data], axis=0)
    viz.violin(ax, data=comb_data, y='gap',
               x='model', order=['human', 'model'],
               palette=[viz.oGrey, color],
               errorbar='se', errorlw=2, err_capsize=.2,
               scatter_edge_color='w', scatter_lw=.2,
               mean_marker_size=5.5, scatter_size=3)
    ax.axhline(y=0, xmin=0, xmax=1, color='k', lw=1, ls='--')
    ax.spines['left'].set_position(('axes',-0.08))
    #for pos in ['bottom','left']: ax.spines[pos].set_linewidth(2)
    ax.set_xlabel('Agent')
    #ax.set_title(eval(model).name)
    ax.set_ylim([-.2, .55])
   
def summarize_model_behaviors(data_set, model, cross_valid=False, figsize=(10, 1.5)):
    plt.rcParams['xtick.labelsize']    = 8.2 
    plt.rcParams['ytick.labelsize']    = 8.2
    sim_data = get_beh_data(data_set, model, cross_valid=cross_valid)
    if model!='human': sim_data['r'] = sim_data['acc']
    fig, axs = plt.subplots(1, 6, figsize=figsize)
    ax = axs[0]
    learning_speed_target(ax, sim_data.copy(), with_target_data_set=data_set)
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1])
    ax.set_yticklabels([.2, '', .6, '', 1.0])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('')
    ax.legend().remove()
    ax.set_box_aspect(1.08)
    ax = axs[1]
    forward_interference_target(ax, sim_data.copy())
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels([1, 6, 11])
    ax.set_ylim([.2, 1.05])
    ax.set_yticks([.2, .4, .6, .8, 1])
    ax.set_yticklabels([.2, '', .6, '', 1.0])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('')
    ax.legend().remove()
    ax.set_box_aspect(1.08)
    ax = axs[2]
    catastrophic_forgetting_target(ax, pred_data=sim_data, 
                tar_data_set=data_set, 
                block_type='block',
                color=viz.Blue)
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    ax.set_box_aspect(1.3)
    ax.set_xlabel('')
    ax.set_ylim([.2, 1.05])
    ax.set_ylabel('')
    ax.set_yticks([.2, .4, .6, .8, 1])
    ax.set_yticklabels([.2, '', .6, '', 1.0])
    ax = axs[3]
    catastrophic_forgetting_target(ax, pred_data=sim_data, 
                tar_data_set=data_set, 
                block_type='interleave',
                color=viz.Red)
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    ax.set_box_aspect(1.25)
    ax.set_ylim([.2, 1.05])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([.2, .4, .6, .8, 1])
    ax.set_yticklabels(['', '', '', '', ''])
    ax = axs[5]
    learn_gap_top50_with_target(ax, sim_data, tar_data_set=data_set)
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    ax.set_box_aspect(1.08)
    ax.set_xlabel('')
    ax = axs[4]
    get_test_per_s_target(ax, sim_data, block_type='block', 
                          tar_data_set=data_set,
                          stimuli=np.arange(4))
    for pos in ['bottom', 'left']: ax.spines[pos].set_linewidth(1.75)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('')
    ax.set_ylim([0, 1.1])
    ax.set_box_aspect(1)
    #fig.tight_layout()

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

def model_compare_group(axs, llh_table, pxp, models):
    ax = axs[0]
    palette = [eval(m).color for m in models]
    viz.violin(data=llh_table, x='BIC', y='model', 
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
    ax.set_xlabel(r'$\Delta$'+f'BIC')
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
     