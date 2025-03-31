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
    block_types = ['block', 'interleave']
    
    def __init__(self, block_type, folder='block0'): 
        '''
        block in ['block', 'interleave']
        '''
        super().__init__(block_type)
        self._init_S(block_type)
    
    def _init_S(self, block_type):
        self.nS = block_type
        
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
