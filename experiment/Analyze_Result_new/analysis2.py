import pandas as pd
import argparse
import numpy as np
import math
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ARIMA', help='The base forecasts method')
parser.add_argument('--dataset', type=str, default='labour', help='The name of dataset')

if __name__ == '__main__':

    # Get params
    args = parser.parse_args()
    method = args.method
    dataset = args.dataset


    df = pd.read_csv(f'./Analyze_Result_new/CRPS/{dataset}_{method}_mean_crps_5.csv')
    
    if dataset == 'Tourism':
        lis = [1,8,35,111]
    else:
        lis = [1,9,57]
    

    num = len(lis)
    df1 = pd.DataFrame()
    name = []
    for i in range(num):
        name.append('level'+str(i+1))
    
    df1[name[0]] = df.iloc[0,1:]
    for i in range(1,num):
        df1[name[i]] = df.iloc[lis[i-1]:lis[i]].mean(axis=0)
    # i += 1
    # df1[name[i]] = df.iloc[lis[i]:].mean(axis=0)
    df1 = df1.round(4)
    df1.to_csv(f'./Analyze_Result_new/output/{dataset}_{method}_mean_level_crps_5.csv')

