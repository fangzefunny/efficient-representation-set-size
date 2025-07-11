import numpy as np 
import pandas as pd 

eps_ = 1e-10

class base_task:
    name = 'base_task'
    voi = ['a', 'acc', 'r', 'confidence']
    block_types = ['cont']

    def __init__(self, block_type):
        self.block_type = block_type
    
    @staticmethod
    def eval_fn(row, subj, **kwargs):
        return NotImplementedError
    
    @staticmethod
    def sim_fn(row, subj, rng, **kwargs):
        return NotImplementedError

    def embed(s):
        return NotImplementedError

    def instan(self, seed=1234, **kwargs):
        return NotImplementedError
    
    def termination(self, block_data, pred_data):
        '''Termination condition for this task 

        However, this is not always used. It often 
        used when the task has adapative termination
        condition. 
        '''
        return False

# --------------------------- #
#       Forgetting task       #
# --------------------------- #
    
class setsize_task(base_task):
    name = 'setsize_task'
    nA   = 3
    n_train_rep = 12
    n_test_rep = 6
    block_types = [2, 3, 4, 5, 6]
    max_rep = 15
    min_rep = 9
    
    def __init__(self, block_type, folder='block0'): 
        '''
        block in ['block', 'interleave']
        '''
        super().__init__(block_type)
        self._init_S(block_type)
    
    def _init_S(self, block_type):
        self.nS = int(block_type)

    def termination(self, block_data, pred_data):
        '''Termination condition for this task 

        The task terminates either:
            (1) the number of trials reaches the 15 maximum repetitions
            (2) reach 9 repetitions & four of the last five blocks are correct
        '''
        block_type = block_data['block_type'].unique()[0]
        min_trials = block_type*self.min_rep
        max_trials = block_type*self.max_rep
        q1 = pred_data.dropna().__len__() >= max_trials
        q2 = pred_data.dropna().__len__() >= min_trials
        q3 = pred_data.dropna()['r'].values[-5:].mean() >= .8
        return q1 or (q2 and q3)
    
    @staticmethod
    def fill_trials(block_data, rng):
        '''Fill the block_data to the maximum number of trials

        This experiment forces to end when the repetition reaches 15.
        '''
        # get the current repetition in the block_data
        current_reps = block_data['repetitions'].max()
        current_trials = block_data['trial'].max()
        fill_reps = setsize_task.max_rep - current_reps
        # get the state space 
        S = block_data['s'].unique().tolist()
        # get the s-a mapping
        s_cor_a_table = block_data.pivot_table(
            index='s',
            values='cor_a',
            aggfunc='mean'
        )
        s_cor_a_map = {s: s_cor_a_table.loc[s].values[0] for s in s_cor_a_table.index}
        # fill the block_data
        fill_block_data = {'trial':[], 's':[], 'cor_a':[], 'repetitions':[]}
        for rep in range(int(fill_reps)):
            rng.shuffle(S)
            cor_a_seq = [s_cor_a_map[s] for s in S]
            # save the data
            fill_block_data['s']+= S.copy()
            fill_block_data['cor_a'] += cor_a_seq.copy()
            fill_block_data['repetitions'] += [current_reps+rep+1]*len(S)
            start_idx = int(current_trials+rep*len(S))
            end_idx = int(start_idx+len(S))
            fill_block_data['trial'] += list(range(start_idx+1, end_idx+1))
        # fill the other information
        cols = ['sub_id', 'block_id', 'a_ava', 'block_type', 'stage']
        for c in cols:
            fill_value = block_data[c].unique()
            assert fill_value.__len__() == 1, f'{c} has multiple values'
            fill_block_data[c] = fill_value[0]
        fill_block_data = pd.DataFrame(fill_block_data)
        filled_data = pd.concat([block_data, fill_block_data], axis=0)
        return filled_data.reset_index(drop=True)
        
    @staticmethod
    def eval_fn(row, subj):
        
        block_type = row['block_type']
        stage = row['stage'] 
        s     = int(row['s'])
        a_ava = eval(row['a_ava'])
        pi    = subj.policy(s=s,  
                            block_type=block_type,
                            stage=stage,
                            a_ava=a_ava)
        a     = int(row['a'])
        r     = row['r']
        ll    = np.log(pi[a]+eps_)

        # save the info and learn
        subj.mem.push({
                's': s,
                'a': a,
                'a_ava': a_ava,
                'r': r,
                'block_type': block_type,
            }) 
        if stage == 'train': subj.learn()

        return ll
    
    @staticmethod
    def sim_fn(row, subj, rng):
        
        block_type = row['block_type']
        stage = row['stage']
        s     = int(row['s'])
        a_ava = eval(row['a_ava'])
        pi    = subj.policy(s=s, 
                            block_type=block_type,
                            stage=stage,
                            a_ava=a_ava)
        pi_sort = np.sort(pi) 
        nA = len(pi)
        a     = int(rng.choice(nA, p=pi))
        cor_a = int(row['cor_a'])
        r     = 1.*(a==cor_a)
        confidence = 2*(r-1/nA)*(pi_sort[-1]-pi_sort[-2]) # (li, 2020)

        # save the info and learn 
        subj.mem.push({
                's': s,
                'a': a,
                'a_ava': a_ava,
                'r': r,
                'block_type': block_type,
            })
        if stage == 'train': subj.learn()

        return a, pi[cor_a].copy(), r, confidence

    def instan(self, max_reps=30, seed=1234, **kwargs):

        # set the seed 
        rng = np.random.RandomState(seed)

        # get the state space
        S = np.arange(self.nS)
        # get the correct action, check if the
        # mapping cover all three actions
        # when set size is not 2.
        map_pass = False
        while not map_pass:
            # sample the correct action
            cor_a = rng.choice(self.nA, size=self.nS, replace=True)
            # check if the mapping cover all threa actions 
            if self.nS == 2:
                if np.unique(cor_a).__len__() == 2: 
                    map_pass = True
            else:
                if np.unique(cor_a).__len__() == self.nA: 
                    map_pass = True
        # state-cor_a mapping 
        s_cor_a_map = {s: cor_a[s] for s in S}

        # initialize the stim_seq 
        cols = ['s', 'cor_a', 'a_ava', 'block_type', 'stage', 'repetitions']
        task_data = {c: [] for c in cols}
        for rep in range(max_reps):
            # shuffle the mini-block 
            rng.shuffle(S)
            task_data['s'].extend(S.copy())
            block_cor_a = [s_cor_a_map[s] for s in S]
            task_data['cor_a'].extend(block_cor_a.copy())
            task_data['a_ava'].extend(["[0,1,2]"]*self.nS)
            task_data['block_type'].extend([self.nS]*self.nS)
            task_data['stage'].extend(['train']*self.nS)
            task_data['repetitions'].extend([rep+1]*self.nS)
        # convert to dataframe
        task_data = pd.DataFrame(task_data)
        task_data['trials'] = np.arange(task_data.shape[0])+1
        return task_data

class uncomp_task(setsize_task):
    name = 'uncomp_task'
    block_types = ['3S3A', '3S10A', '10S3A', '10S10A']

    def __init__(self, block_type, folder='block0'):
        self.block_type = block_type
        self._init_S(block_type)
    
    def _init_S(self, block_type):
        self.nS = int(block_type.split('S')[0])
        self.nA = int(block_type.split('S')[1].split('A')[0])
    
    def termination(self, block_data, pred_data):
        '''Termination condition for this task 

        The task terminates either:
            (1) the number of trials reaches the 30 maximum repetitions
            (2) reach 15 repetitions & 10 of the last 9 blocks are correct
        '''
        block_type = block_data['block_type'].unique()[0]
        min_trials = block_type*15
        max_trials = block_type*30
        q1 = pred_data.dropna().__len__() >= max_trials
        q2 = pred_data.dropna().__len__() >= min_trials
        q3 = pred_data.dropna()['r'].values[-9:].mean() >= .8
        return q1 or (q2 and q3)
    
    def instan(self, max_reps=30, seed=1234, **kwargs):

        # set the seed 
        rng = np.random.RandomState(seed)

        # get the state space
        S = np.arange(self.nS)
        # define the correct action 
        if self.nS == self.nA:
            cor_a = np.arange(self.nA)
            rng.shuffle(cor_a)
        else:
            cor_a = rng.choice(self.nA, size=self.nS, replace=True)
        
        # state-cor_a mapping 
        s_cor_a_map = {s: cor_a[s] for s in S}

        # initialize the stim_seq 
        cols = ['s', 'cor_a', 'a_ava', 'block_type', 'stage', 'repetitions']
        task_data = {c: [] for c in cols}
        for rep in range(max_reps):
            # shuffle the mini-block 
            rng.shuffle(S)
            task_data['s'].extend(S.copy())
            block_cor_a = [s_cor_a_map[s] for s in S]
            task_data['cor_a'].extend(block_cor_a.copy())
            a_ava = "[0,1,2]" if self.nA==3 else "[0,1,2,3,4,5,6,7,8,9]"
            task_data['a_ava'].extend([a_ava]*self.nS)
            task_data['block_type'].extend([self.block_type]*self.nS)
            task_data['stage'].extend(['train']*self.nS)
            task_data['repetitions'].extend([rep+1]*self.nS)
        # convert to dataframe
        task_data = pd.DataFrame(task_data)
        task_data['trials'] = np.arange(task_data.shape[0])+1
        return task_data
    
  