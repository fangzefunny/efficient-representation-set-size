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

# --------------------------- #
#       Forgetting task       #
# --------------------------- #
    
class setsize_task(base_task):
    name = 'setsize_task'
    nA   = 3
    n_train_rep = 12
    n_test_rep = 6
    block_types = [2, 3, 4, 5, 6]
    
    def __init__(self, block_type, folder='block0'): 
        '''
        block in ['block', 'interleave']
        '''
        super().__init__(block_type)
        self._init_S(block_type)
    
    def _init_S(self, block_type):
        self.nS = int(block_type)
        
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
        a     = int(rng.choice(setsize_task.nA, p=pi))
        cor_a = int(row['cor_a'])
        r     = 1.*(a==cor_a)
        confidence = 2*(r-1/setsize_task.nA)*(pi_sort[-1]-pi_sort[-2]) # (li, 2020)

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
