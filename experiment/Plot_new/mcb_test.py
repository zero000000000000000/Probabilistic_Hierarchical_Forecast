import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ETS', help='The base forecasts method')
parser.add_argument('--dataset', type=str, default='Tourism', help='The name of dataset')

plt.rcParams.update({'font.size': 20})

if __name__=='__main__':
    # Get params
    args = parser.parse_args()
    method = args.method
    dataset = args.dataset

    df = pd.read_csv(f'./Analyze_Result_new/CRPS/{dataset}_{method}_mean_crps.csv').iloc[:,1:]

    if method == 'deepar':
        df.columns = ['Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','EnergyScore_Opt']
        k = 8
        method = 'DeepAR'
    
    else:
        df.columns = ['Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','MinTShr','MinTShrv','EnergyScore_Opt']
        k = 10
    
    if dataset == 'Tourism':
        N = 111
    else:
        N = 199

    rank_res = df.rank(axis=1)
    r = 4.743
    d = k*(k+1)/(12*N)
    x = rank_res.mean().sort_values()
    y = list(range(0,k,1))
    xerr = d**0.5*r

    # plot:
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('white')
    ax.errorbar(x, y, xerr = xerr, fmt='o', linewidth=2, capsize=6)
    ax.set_yticks(list(range(0,k,1)))
    ax.set_yticklabels(list(x.index))
    ax.set_title(f'{dataset}::{method}')
    plt.tight_layout()
    plt.savefig(f'./Plot_new/{dataset}/MCB_Test_{method}.png')
    plt.show()