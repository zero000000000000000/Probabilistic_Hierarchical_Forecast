import pandas as pd
import argparse
import numpy as np
import math
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='ETS', help='The base forecasts method')
parser.add_argument('--dataset', type=str, default='Tourism', help='The name of dataset')
parser.add_argument('--num',type=int,default=1)

if __name__ == '__main__':

    # Get params
    args = parser.parse_args()
    method = args.method
    dataset = args.dataset
    num1 = args.num

    df = pd.read_csv(f'./Evaluation_Result_new/Opt/{dataset}_{method}_{num1}_c.csv')
    df = df.groupby('series').mean()
    df.reset_index(inplace=True)
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
        df1[name[i]] = df.iloc[lis[i-1]:lis[i],1:].mean(axis=0)
    # i += 1
    # df1[name[i]] = df.iloc[lis[i]:].mean(axis=0)
    df1 = df1.round(2)
    df1.to_csv(f'./Analyze_Result_new/output_opt/{dataset}_{method}_{num1}.csv')
    df_mean = df1.mean(axis=0).round(2)
    df_sd = df1.std(axis=0,ddof=1).round(2)
    df_mean.to_csv(f'./Analyze_Result_new/output_opt/{dataset}_{method}_{num1}_mean.csv')
    df_sd.to_csv(f'./Analyze_Result_new/output_opt/{dataset}_{method}_{num1}_sd.csv')



