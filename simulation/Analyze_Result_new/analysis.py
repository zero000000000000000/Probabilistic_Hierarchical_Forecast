import numpy as np
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt

colors = ["#6A5ACD", "#708090", "#20B2AA", "#FFA07A", "#ADD8E6", "#8B4513",
          "#BA55D3", "#90EE90", "#FFD700", "#FF6347"]

parser = argparse.ArgumentParser()
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')
parser.add_argument('--basefdep', type=str, default='Independent', help='The base forecast independence')

def plot_avg_energy_score(data,path1,path):
    '''
    Plot the line of average energy score with different reconciliation method
    '''
    global colors
    #print(data.head())
    data = data.iloc[:,1:]
    data = data.mean()

    # Save the processed data
    data.to_csv(path1+'_mean_es_v2.csv',index=True)

    ax = data.plot(kind='bar',
                 title='Average Energy Score',
                 color=colors,
                 figsize=(12, 9))
    ax.set_xlabel('Reconciliation Method')
    ax.set_ylabel('Energy Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.1, format(p.get_height(), '.2f'),
                ha='center', va='bottom', fontsize=12)
    plt.savefig(path+'_avg_v2.png')
    #plt.show()
    plt.close()

def plot_avg_variogram_score(data,path1,path):
    '''
    Plot the line of average energy score with different reconciliation method
    '''
    global colors
    #print(data.head())
    data = data.iloc[:,1:]
    data = data.mean()

    # Save the processed data
    data.to_csv(path1+'_mean_vs_v2.csv',index=True)

    ax = data.plot(kind='bar',
                 title='Average Variogram Score',
                 color=colors,
                 figsize=(12, 9))
    ax.set_xlabel('Reconciliation Method')
    ax.set_ylabel('Variogram Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.1, format(p.get_height(), '.2f'),
                ha='center', va='bottom', fontsize=12)
    plt.savefig(path+'_avg_v2.png')
    #plt.show()
    plt.close()

def plot_avg_crps(data,path1,path):
    '''
    Plot the line of average crps with different reconciliation method and time step
    '''
    global colors
    #print(data.head())
    data = data.iloc[:,1:]
    data = data.groupby('series').mean()

    # Save the processed data
    data.to_csv(path1+'_mean_crps_v2.csv',index=True)

    ax = data.plot(kind='line',
                   title='Average CRPS',
                   grid=True,
                   color=colors,
                   legend=True,
                   figsize=(12, 9))
    ax.set_xlabel('Series Index')
    ax.set_ylabel('CRPS')
    plt.savefig(path+'_avg_v2.png')
    #plt.show()
    plt.close()

def integration(opt_data):
    '''
    Integrate the results
    '''
    opt_crps = opt_data['CRPS']
    opt_crps = [item for times in opt_crps for item in times]

    opt_es = opt_data['ES']
    opt_es = [item for times in opt_es for item in times]

    opt_vs = opt_data['VS']
    return [opt_crps,opt_es,opt_vs]


if __name__=='__main__':

    # Get params
    args = parser.parse_args()
    generate = args.generate
    rootbasef = args.rootbasef
    basefdep = args.basefdep

    # Get R results
    r_es_data = pd.read_csv(f'./Evaluation_Result_new/Energy_Score/{generate}_{rootbasef}_{basefdep}_v2.csv')
    r_crps_data = pd.read_csv(f'./Evaluation_Result_new/CRPS/{generate}_{rootbasef}_{basefdep}_v2.csv')
    r_vs_data = pd.read_csv(f'./Evaluation_Result_new/Variogram_Score/{generate}_{rootbasef}_{basefdep}_v2.csv')


    plot_avg_energy_score(r_es_data,
                          path1=f'./Analyze_Result_new/Energy_Score/{generate}_{rootbasef}_{basefdep}',
                          path=f'./Plot_new/Energy_Score/{generate}_{rootbasef}_{basefdep}')
    plot_avg_variogram_score(r_vs_data,
                             path1=f'./Analyze_Result_new/Variogram_Score/{generate}_{rootbasef}_{basefdep}',
                             path=f'./Plot_new/Variogram_Score/{generate}_{rootbasef}_{basefdep}')
    plot_avg_crps(r_crps_data,
                  path1=f'./Analyze_Result_new/CRPS/{generate}_{rootbasef}_{basefdep}',
                  path=f'./Plot_new/CRPS/{generate}_{rootbasef}_{basefdep}')