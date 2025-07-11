import pingouin as pg 

def t_test(x_data, y_data, paired=False, alternative='two-sided', title=''):
    df = pg.ttest(x_data, y_data, paired=paired, alternative=alternative)
    dof = df.loc[:, 'dof'].values[0]
    t   = df.loc[:, 'T'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    cohen_d = df.loc[:, 'cohen-d'].values[0]
    pair_str = '-paired' if paired==True else ''
    print(f'{title} \tt{pair_str}({dof:.3f})={t:.3f}, p={pval:.3f}, cohen-d={cohen_d:.3f}')

def mwu(x_data, y_data, alternative='two-sided', title=''):
    df = pg.mwu(x_data, y_data, alternative=alternative)
    n = df.shape[0]
    U  = df.loc[:, 'U-val'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    CLES = df.loc[:, 'CLES'].values[0]
    print(f'{title} \tU({n})={U:.3f}, p={pval:.3f}, CLES={CLES:.3f}')

def wilcoxon(x_data, y_data, alternative='two-sided', title=''):
    df = pg.wilcoxon(x_data, y_data, alternative=alternative)
    n = x_data.shape[0]
    W  = df.loc[:, 'W-val'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    CLES = df.loc[:, 'CLES'].values[0]
    print(f'{title} \tU({n})={W:.3f}, p={pval:.3f}, CLES={CLES:.3f}')

def corr(x_data, y_data, title='', method='pearson'):
    df = pg.corr(x_data, y_data, method=method)
    n = df.loc[:, 'n'].values[0]
    r   = df.loc[:, 'r'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    print(f'{title} \tr({n})={r:.3f}, p={pval:.3f}')

def anova(dv, between, data, all_table=False):
    df = pg.anova(dv=dv, between=between, data=data).rename(columns={'p-unc': 'punc'})
    sig_df = df if all_table else df.query('punc<=.05') 
    for _, row in sig_df.iterrows():
        title = row['Source']
        dof1  = int(row['ddof1'])
        dof2  = int(row['ddof2'])
        F     = row['F']
        p     = row['punc']
        np2   = row['np2']
        print(f'\t{title}:\tF({dof1}, {dof2})={F:.3f}, p={p:.3f}, np2={np2:.3f}')
    if not all_table:
        other_df = df.query('punc>.05')
        if other_df.shape[0] > 0:
            other_min_p = other_df['punc'].min()
            print(f'\tOther: \tp>={other_min_p:.3f}')

def linear_regression(x, y, add_intercept=False, title='', x_var='x', y_var='y'):
    df = pg.linear_regression(X=x, y=y, add_intercept=add_intercept)
    beta0 = df['coef'][0]
    beta1 = df['coef'][1]
    pval  = df['pval'][1]
    print(f'{title}\t{y_var}={beta1:.3f}{x_var}+{beta0:.3f},\n\tp={pval:.3f}')
