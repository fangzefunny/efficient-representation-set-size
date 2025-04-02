import os 
import numpy as np 
import pandas as pd 
from copy import deepcopy

from functools import lru_cache
from scipy.special import softmax 
from scipy.stats import halfnorm, uniform

from utils.fit import *
from utils.env_fn import *
from utils.viz import *

pth = os.path.dirname(os.path.abspath(__file__))

eps_ = 1e-13
max_ = 1e+13

# ------------------------------#
#          Axuilliary           #
# ------------------------------#

def mask_fn(nA, a_ava):
    return (np.eye(nA)[a_ava, :]).sum(0, keepdims=True)

def MI(p_X, p_Y1X, p_Y):
    return (p_X*p_Y1X*(np.log(p_Y1X+eps_)-np.log(p_Y.T+eps_))).sum()

def clip_exp(x):
    x = np.clip(x, a_min=-max_, a_max=50)
    return np.exp(x) 

def step(w, lr):
    w.data -= lr*w.grad.data
    w.grad.data.zero_()

# ------------------------------#
#         Agent wrapper         #
# ------------------------------#

class wrapper:
    '''Agent wrapper

    We use the wrapper to

        * Fit
        * Simulate
        * Evaluate the fit 
    '''

    def __init__(self, agent, env_fn):
        self.agent  = agent
        self.env_fn = env_fn
        self.use_hook = False
    
    # ------------ fit ------------ #

    def fit(self, data, method, alg, pool=None, p_priors=None,
            init=False, seed=2021, verbose=False, n_fits=40):
        '''Fit the parameter using optimization 
        '''

        # get functional inputs 
        fn_inputs = [self.loss_fn, 
                     data, 
                     self.agent.p_bnds,
                     self.agent.p_pbnds, 
                     self.agent.p_names,
                     self.agent.p_priors if p_priors is None else p_priors,
                     method,
                     alg, 
                     init,
                     seed,
                     verbose]
        
        if pool:
            sub_fit = fit_parallel(pool, *fn_inputs, n_fits=n_fits)
        else: 
            sub_fit = fit(*fn_inputs)  

        return sub_fit      

    def loss_fn(self, params, sub_data, p_priors=None):
        '''Total likelihood

        Fit individual:
            Maximum likelihood:
            log p(D|θ) = log \prod_i p(D_i|θ)
                       = \sum_i log p(D_i|θ )
            or Maximum a posterior 
            log p(θ|D) = \sum_i log p(D_i|θ ) + log p(θ)
        '''
        # negative log likelihood
        tot_loglike_loss  = -np.sum([self.loglike(params, sub_data[key])
                    for key in sub_data.keys()])
        # negative log prior 
        if p_priors==None:
            tot_logprior_loss = 0 
        else:
            p_trans = [fn(p) for p, fn in zip(params, self.agent.p_trans)]
            tot_logprior_loss = -self.logprior(p_trans, p_priors)
        # sum
        return tot_loglike_loss + tot_logprior_loss

    def loglike(self, params, block_data):
        '''Likelihood for one sample
        -log p(D_i|θ )
        In RL, each sample is a block of experiment,
        Because it is independent across experiment.
        '''
        # init subject and load block type
        block_type_lst = block_data['block_type'].unique()
        assert len(block_type_lst) == 1, 'Only one block type is allowed'
        block_type = block_type_lst[0]
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)
        ll   = 0
       
        ## loop to simulate the responses in the block 
        for t, row in block_data.iterrows():

            # predict stage: obtain input
            ll += env.eval_fn(row, subj)

        return ll
          
    def logprior(self, params, p_priors):
        '''Add the prior of the parameters
        '''
        lpr = 0
        for pri, param in zip(p_priors, params):
            lpr += np.max([pri.logpdf(param), -max_])
        return lpr

    # ------------ evaluate ------------ #

    def eval(self, data, params):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            sim_data.append(self.eval_block(block_data, params))
        return pd.concat(sim_data, ignore_index=True)
    
    def eval_block(self, block_data, params):

        # init subject and load block type
        block_type_lst = block_data['block_type'].unique()
        assert len(block_type_lst) == 1, 'Only one block type is allowed'
        block_type = block_type_lst[0]
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = ['ll'] + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # record some insights of the model
            for v in self.agent.voi:
                pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # simulate the data 
            ll = env.eval_fn(row, subj)
            
            # record the stimulated data
            pred_data.loc[t, 'll'] = ll

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)

    # ------------ simulate ------------ #

    def sim(self, data, params, rng):
        sim_data = [] 
        for block_id in data.keys():
            block_data = data[block_id].copy()
            for v in self.env_fn.voi:
                if v in block_data.columns:
                    block_data = block_data.drop(columns=v)
            sim_data.append(self.sim_block(block_data, params, rng))
        
        return pd.concat(sim_data, ignore_index=True)

    def sim_block(self, block_data, params, rng):

        # init subject and load block type
        block_type_lst = block_data['block_type'].unique()
        assert len(block_type_lst) == 1, 'Only one block type is allowed'
        block_type = block_type_lst[0]
        env  = self.env_fn(block_type)
        subj = self.agent(env, params)

        ## init a blank dataframe to store variable of interest
        col = self.env_fn.voi + self.agent.voi
        init_mat = np.zeros([block_data.shape[0], len(col)]) + np.nan
        pred_data = pd.DataFrame(init_mat, columns=col)  

        ## loop to simulate the responses in the block
        for t, row in block_data.iterrows():

            # record some insights of the model
            for i, v in enumerate(self.agent.voi):
                pred_data.loc[t, v] = eval(f'subj.get_{v}()')

            # if register hook to get the model insights
            if self.use_hook:
                for k in self.insights.keys():
                    self.insights[k].append(eval(f'subj.get_{k}()'))

            # simulate the data 
            subj_voi = env.sim_fn(row, subj, rng)
            
            # record the stimulated data
            for i, v in enumerate(env.voi): 
                pred_data.loc[t, v] = subj_voi[i]

        # drop nan columns
        pred_data = pred_data.dropna(axis=1, how='all')
            
        return pd.concat([block_data, pred_data], axis=1)
    
    def register_hooks(self, *args):
        self.use_hook = True 
        self.insights = {k: [] for k in args}

# ------------------------------#
#         Memory buffer         #
# ------------------------------#

class simpleBuffer:
    '''Simple Buffer 2.0
    Update log: 
        To prevent naive writing mistakes,
        we turn the list storage into dict.
    '''
    def __init__(self):
        self.m = {}
        
    def push(self, m_dict):
        self.m = {k: m_dict[k] for k in m_dict.keys()}
        
    def sample(self, *args):
        lst = [self.m[k] for k in args]
        if len(lst)==1: return lst[0]
        else: return lst

# ------------------------------#
#             Base              #
# ------------------------------#

class base_agent:
    '''Base Agent'''
    name     = 'base'
    p_bnds   = None
    p_pbnds  = []
    p_names  = []  
    p_priors = []
    p_trans  = []
    p_links  = []
    n_params = 0 
    # value of interest, used for output
    # the interesting variable in simulation
    voi      = []
    insights = ['pol']
    
    def __init__(self, env, params):
        self.env = env 
        self.nS  = env.nS
        self.nA  = env.nA 
        self.load_params(params)
        self._init_embed()
        self._init_buffer()
        self._init_agent()
        
    def load_params(self, params): 
        return NotImplementedError
    
    def _init_embed(self):
        self.embed = self.env.embed

    def _init_buffer(self):
        self.mem = simpleBuffer()
    
    def _init_agent(self):
        return NotImplementedError
    
    def learn(self): 
        return NotImplementedError

    def policy(self, s, **kwargs): 
        return NotImplementedError

    # --------- Insights -------- #

    def get_pol(self):
        acts = [[0, 1], [2, 3]]
        pi = np.zeros([self.nS, self.nA])
        for s in range(self.nS):
            for act in acts:
                a1, a2 = act
                pi[s, :] += self.policy(self.s2f(s), a_ava=[a1, a2])   
        return pi

class human:
   name  = 'Human'
   color = viz.Blue 

def loss_capacity(theta, c_tar, nS):
        p_x = np.ones([nS, 1])/nS
        p_y1x = softmax(theta*np.eye(nS), axis=1)
        p_y = p_y1x@p_x
        c_hat = (p_x*p_y1x*(np.log(p_y1x+1e-16)-np.log(p_y.T+1e-16))).sum()
        return (c_tar-c_hat)**2

@lru_cache(maxsize=None)
def theta_given_C(c_tar, nS):
        theta0  = np.random.rand()*3
        res = minimize(loss_capacity, x0=theta0, args=(c_tar, nS))
        theta = res.x[0]
        return theta

class ecPG(base_agent):
    '''Efficient coding policy gradient (analytical)

    We create two mathematical equivalent versions for ECPG, each
    aim at tackling different numerical problems. 
    The only difference exist in calculating sTheta.

    The analytical version, we do:
        sTheta = (u*p_a1Z.T - self.lmbda*log_dif)
    and in the fitting version, we do:
        sTheta = (u*p_a1Z.T/(self.lmdba+eps_) - log_dif)

    The analytical version is more consistent with the objective function.
    It is easy to illustrate the model behaviors with varying
    λ, because it will not divided by 0. 

    However, this implementation causes high correlation
    between parameters α_ψ and λ，bring difficulties in parameter
    estimation. 

    The same logics applied to fECPG.
    '''
    name     = 'ECPG'
    p_names  = ['alpha_psi', 'alpha_rho', 'lmbda', 'capacity']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 3), (-2, 3), (-6, 1.5), (-.22, .26)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*len(p_names)
    p_trans  = [lambda x: clip_exp(x)]*len(p_names)
    p_links  = [lambda x: np.log(x+eps_)]*len(p_names)
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['enc', 'dec', 'pol']
    color    = viz.Red
    marker   = '^'
    size     = 125
    alpha    = 1

    @staticmethod
    def link_params(params):
        return [f(p) for f, p in zip(ecPG.p_links, params)]

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.lmbda      = params[2]
        self.C          = params[3]
        self.b          = 1/3

    def _init_agent(self):
        self.nS = int(self.nS)
        self.nZ = self.nS
        theta = deepcopy(theta_given_C(self.C, self.nS))
        self.theta = np.eye(self.nS)*theta
        self.phi   = np.zeros([self.nS, self.nA]) 
        self._learn_pZ()

    def _learn_pZ(self):
        self.F = np.vstack([np.eye(self.nS)[s] for s in range(self.nS)])
        self.p_Z1S = softmax(self.F@self.theta, axis=1)
        self.p_S   = np.ones([self.nS, 1]) / self.nS  # nSx1 
        self.p_Z   = self.p_Z1S.T @ self.p_S  

    def policy(self, **kwargs):
        f = np.eye(self.nS)[kwargs['s']].reshape([1, -1])
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, kwargs['a_ava'])
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        # renormalize to avoid numeric problem
        pi = (p_Z1s@p_A1Z).reshape([-1])
        return pi / pi.sum()
        
    def learn(self):
        self._learn_enc_dec()
        self._learn_pZ()

    def _learn_enc_dec(self):
        # get data 
        s, a_ava, a, r = self.mem.sample('s', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = np.eye(self.nS)[s].reshape([1, -1])
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        # note: in derviation we wrote 
        #          log_dif = log p(Z|S) - log p(Z) - 1
        # However, substracting the constant 1 will not affect the 
        # numerical value of gradeint, due to the normalization 
        # term in calculating gTheta.
        log_dif = np.log(p_Z1s+eps_)-np.log(self.p_Z.T+eps_)  
        sTheta = (u*p_a1Z.T/(self.lmbda+eps_) - log_dif)
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        sPhi = u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi
    
    # --------- some predictions ----------- #
        
    def get_i_SZ(self):
        psi_Z1S = softmax(self.F@self.theta, axis=1)
        return MI(self.p_S, psi_Z1S, self.p_Z)
    
    def get_i_ZA(self):
        rho_A1Z = softmax(self.phi, axis=1)
        p_A = (self.p_Z.T @ rho_A1Z).T
        return MI(self.p_Z, rho_A1Z, p_A)
        
    def get_enc(self):
        return softmax(self.F@self.theta, axis=1)
    
    def get_dec(self):
        acts = [[0, 1], [2, 3]]
        rho  = 0
        for act in acts:
            m_A   = mask_fn(self.nA, act)
            p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
            p_A1Z  /= p_A1Z.sum(1, keepdims=True)
            rho += p_A1Z
        return rho 
    
class ECRL(ecPG):
    name     = 'ECRL'
    p_names  = ['alpha_psi', 'alpha_rho', 'lmbda', 'capacity', 'alpha_w']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 3), (-2, 3), (-6, 1.5), (-.22, .7), (-3, 1)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*3+\
                [uniform(0, 1.8)]+\
                [halfnorm(0, 40)]
    p_trans  = [lambda x: clip_exp(x)]*3+\
                [lambda x: 1.8/(1+clip_exp(-x))]+\
                [lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_)]*3+\
                [lambda x: np.log(x+eps_)-np.log(1.8-x+eps_)]+\
                [lambda x: np.log(x+eps_)]
    n_params = len(p_names)
    voi      = ['i_SZ']
    insights = ['enc', 'dec', 'pol']
    color    = viz.Red
    marker   = '^'
    size     = 125
    alpha    = 1

    @staticmethod
    def link_params(params):
        return [f(p) for f, p in zip(ECRL.p_links, params)]

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_psi  = params[0]
        self.alpha_rho  = params[1]
        self.lmbda      = params[2]
        self.C          = params[3]
        self.alpha_w    = params[4]
        self.b          = 1/self.nA
        # initialize the w = .9, since
        # w = 1/(1+exp(-xi)), 
        # we have xi = log(.9) - log(.1)
        self.xi         = np.log(.9)-np.log(.1) 
        self.w          = 1/(1+clip_exp(-self.xi))
        
    def _learn_enc_dec(self):
        # get data 
        s, a_ava, a, r = self.mem.sample('s', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = np.eye(self.nS)[s].reshape([1, -1])
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
        
        # backward
        # note: in derviation we wrote 
        #          log_dif = log p(Z|S) - log p(Z) - 1
        # However, substracting the constant 1 will not affect the 
        # numerical value of gradeint, due to the normalization 
        # term in calculating gTheta.
        log_dif = np.log(p_Z1s+eps_)-np.log(self.p_Z.T+eps_)  
        sTheta = (self.w*u*p_a1Z.T/(self.lmbda+eps_) - log_dif)
        gTheta = -f.T@(p_Z1s*(np.ones([1, self.nZ])*
                    sTheta - p_Z1s@sTheta.T))
       
        # calculate the gradient of phi 
        sPhi = self.w*u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        # calculate the gradient of xi 
        gXi = -((p_Z1s@p_a1Z).sum() - 1/self.nA)*(r - self.b)*self.w*(1-self.w)

        # update phi, the parameter of decoder 
        self.theta -= self.alpha_psi * gTheta
        self.phi   -= self.alpha_rho * gPhi 
        self.xi    -= self.alpha_w * gXi

        # update w, the weight for exploration 
        self.w = 1/(1+clip_exp(-self.xi))

class CURL(ECRL):
    name     = 'CURL'
    p_names  = ['alpha_rho', 'alpha_w']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 3), (-3, 1)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*2
    p_trans  = [lambda x: clip_exp(x)]*2
    p_links  = [lambda x: np.log(x+eps_)]*2
    n_params = len(p_names)
    voi      = [] 

    @staticmethod
    def link_params(params):
        return [f(p) for f, p in zip(CURL.p_links, params)]

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_rho  = params[0]
        self.alpha_w    = params[1]
        self.b          = 1/self.nA
        self.xi         = np.log(.9)-np.log(.1) 
        self.w          = 1/(1+clip_exp(-self.xi))

    def _init_agent(self):
        self.nS = int(self.nS)
        self.nZ = self.nS
        theta = 100*np.eye(self.nS)
        self.theta = np.eye(self.nS)*theta
        self.phi   = np.zeros([self.nS, self.nA]) 
        self._learn_pZ()

    def _learn_enc_dec(self):
        # get data 
        s, a_ava, a, r = self.mem.sample('s', 'a_ava', 'a', 'r')
       
        # prediction 
        f     = np.eye(self.nS)[s].reshape([1, -1])
        p_Z1s = softmax(f@self.theta, axis=1)
        m_A   = mask_fn(self.nA, a_ava)
        p_A1Z = softmax(self.phi-(1-m_A)*max_, axis=1)
        p_a1Z = p_A1Z[:, [a]]
        u = np.array([r - self.b])[:, np.newaxis] 
       
        # calculate the gradient of phi 
        sPhi = self.w*u*p_Z1s.T
        gPhi = -p_a1Z*(np.eye(self.nA)[[a]] - p_A1Z)*sPhi

        # calculate the gradient of xi 
        gXi = -((p_Z1s@p_a1Z).sum() - 1/self.nA)*(r - self.b)*self.w*(1-self.w)

        # update phi, the parameter of decoder 
        self.phi   -= self.alpha_rho * gPhi 
        self.xi    -= self.alpha_w * gXi

        # update w, the weight for exploration 
        self.w = 1/(1+clip_exp(-self.xi))

class CCRL(ECRL):
    '''Capacity-constrained RL'''
    name     = 'CCRL'
    p_names  = ['alpha_rho', 'alpha_w', 'C']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(-2, 3), (-3, 1), (-.22, .7)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40)]*2+\
                [uniform(0, 1.8)]
    p_trans  = [lambda x: clip_exp(x)]*2+\
                [lambda x: 1.8/(1+clip_exp(-x))]
    p_links  = [lambda x: np.log(x+eps_)]*2+\
                [lambda x: np.log(x+eps_)-np.log(1.8-x+eps_)]
    n_params = len(p_names)
    voi      = [] 

    @staticmethod
    def link_params(params):
        return [f(p) for f, p in zip(CCRL.p_links, params)]

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.alpha_rho  = params[0]
        self.C          = params[1]
        self.alpha_w    = params[2]
        self.b          = 1/self.nA
        # initialize the w = .9, since
        # w = 1/(1+exp(-xi)), 
        # we have xi = log(.9) - log(.1)
        self.xi         = np.log(.9)-np.log(.1) 
        self.w          = 1/(1+clip_exp(-self.xi))

    def _learn_enc_dec(self): CURL._learn_enc_dec(self)

# --------- RLWM as a baseline --------- #

class RLWM(base_agent):
    name     = 'RLWM'
    p_names  = ['beta_rl', 'alpha_rl', 'eps', 'C', 'w0', 'beta_wm']  
    p_bnds   = [(-1000, 1000)]*len(p_names)
    p_pbnds  = [(1, 2), (-2, 2), (-2, 2), (1, 2), (-2, 2), (1, 2)]
    p_poi    = p_names
    p_priors = [halfnorm(0, 40), uniform(0, 1), uniform(0, 1), 
                halfnorm(0, 40), uniform(0, 1), halfnorm(0, 40)]
    p_trans  = [lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x),
                lambda x: 1/(1+clip_exp(-x)),
                lambda x: clip_exp(x)]
    p_links  = [lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)-np.log(1-x+eps_),
                lambda x: np.log(x+eps_)-np.log(1-x+eps_),
                lambda x: np.log(x+eps_),
                lambda x: np.log(x+eps_)-np.log(1-x+eps_),
                lambda x: np.log(x+eps_)]
    n_params = len(p_names)
    voi      = [] 

    @staticmethod
    def link_params(params):
        return [f(p) for f, p in zip(RLWM.p_links, params)]

    def load_params(self, params):
        params = [f(p) for f, p in zip(self.p_trans, params)]
        self.beta_rl  = params[0]
        self.alpha_rl = params[1]
        self.eps      = params[2]
        self.C        = params[3]
        self.w        = params[4]
        self.beta_wm  = params[5]

    def _init_agent(self):
        # initialize the Q table for WM
        self.qWM_SA = np.ones([self.nS, self.nA]) / self.nA
        # initialize the Q table for RL
        self.qRL_SA = np.ones([self.nS, self.nA]) / self.nA
        # initialize the weight for WM
        self.w *= np.min([1, self.C / self.nS])
        # initialize the weight for RL
        self.ws = np.ones([self.nS]) * self.w

    def learn(self):
        # get data 
        s, a, r = self.mem.sample('s', 'a', 'r')

        # get the Q values
        qWM_hat = self.qWM_SA[s, a]
        qRL_hat = self.qRL_SA[s, a]

        # decay of the working memory to chance level
        self.qWM_SA[s, a] += self.eps*(1/self.nA - self.qWM_SA[s, a])
        # update working memory and create working memory policy
        # get q_wm_t+1
        self.qWM_SA[s, a] += 1 * (r - qWM_hat)


        # update the RL critic, and wokring memory
        # get q_rl_t+1 
        self.qRL_SA[s, a] += self.alpha_rl * (r - qRL_hat)
        
        # update w: get w_t+1
        wc =  np.min([1, self.C/self.nS])
        p_RL = qRL_hat**r*(1-qRL_hat)**(1-r)
        p_WM = wc * qWM_hat**r*(1-qWM_hat)**(1-r) + (1-wc)/self.nA   
        self.ws[s] = (self.ws[s]*p_WM) / (self.ws[s]*p_WM + (1-self.ws[s])*p_RL)

    def policy(self, **kwargs):
        s = kwargs['s']
        pi_RL = softmax(self.beta_rl * self.qRL_SA[s,:])
        pi_WM = softmax(self.beta_wm * self.qWM_SA[s,:])
        return self.ws[s]*pi_WM + (1 - self.ws[s])*pi_RL
