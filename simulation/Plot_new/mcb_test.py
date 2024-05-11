import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser()
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')
parser.add_argument('--basefdep', type=str, default='Independent', help='The base forecast independence')

config = {
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)


#plt.rcParams['font.sans-serif'] = font_list  # 或者使用 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 20

if __name__ == '__main__':

    # Get params
    args = parser.parse_args()
    generate = args.generate
    rootbasef = args.rootbasef
    basefdep = args.basefdep

    df = pd.read_csv(f'./Analyze_Result_new/CRPS/{generate}_{rootbasef}_{basefdep}_mean_crps_v2.csv').iloc[:,1:]
    df.columns = ['Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','MinTShr','MinTShrv','EnergyScore_Opt']
    rank_res = df.rank(axis=1)
    k=10
    N=7
    r = 4.743
    d = k*(k+1)/(12*N)
    x = rank_res.mean().sort_values()
    y = list(range(0,k,1))
    xerr = d**0.5*r

    # plot:
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor('white')
    ax.errorbar(x, y, xerr = xerr, fmt='o', linewidth=2, capsize=6)
    ax.set_yticks(list(range(0,k,1)))
    ax.set_yticklabels(list(x.index),fontproperties='Times New Roman')
    ax.set_title('MCB检验::'+f'{generate}_{rootbasef}_{basefdep}')
    plt.tight_layout()
    plt.savefig(f'./Plot_new/MCB_Test/MCB_Test_{generate}_{rootbasef}_{basefdep}_v2.png')
    # plt.show()