import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

colors = ["#6A5ACD", "#708090", "#20B2AA", "#FFA07A", "#ADD8E6", "#8B4513",
          "#BA55D3", "#90EE90", "#FFD700", "#FF6347", "#00CED1","#FF4500",'blue']


def plot_avg_energy_score(data,path1,path):
    '''
    Plot the line of average energy score with different reconciliation method
    '''
    global colors
    #print(data.head())
    # data = data.iloc[:,:]
    data = data.mean()

    # Save the processed data
    data.to_csv(path1+'_mean_es.csv',index=True)

    ax = data.plot(kind='bar',
                 title='Average Energy Score',
                 color=colors[:10],
                 figsize=(12, 9))
    ax.set_xlabel('Reconciliation Method')
    ax.set_ylabel('Energy Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.1, format(p.get_height(), '.2f'),
                ha='center', va='bottom', fontsize=12)
    plt.savefig(path+'_avg.png')
    #plt.show()
    plt.close()

def plot_avg_variogram_score(data,path1,path):
    '''
    Plot the line of average energy score with different reconciliation method
    '''
    global colors
    #print(data.head())
    # data = data.iloc[:,:]
    data = data.mean()

    # Save the processed data
    data.to_csv(path1+'_mean_vs.csv',index=True)

    ax = data.plot(kind='bar',
                 title='Average Variogram Score',
                 color=colors[:10],
                 figsize=(12, 9))
    ax.set_xlabel('Reconciliation Method')
    ax.set_ylabel('Variogram Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2, p.get_height() + 0.1, format(p.get_height(), '.2f'),
                ha='center', va='bottom', fontsize=12)
    plt.savefig(path+'_avg.png')
    #plt.show()
    plt.close()

def plot_avg_crps(data,path1,path):
    '''
    Plot the line of average crps with different reconciliation method and time step
    '''
    global colors
    #print(data.head())
    # data = data.iloc[:,:]
    data = data.groupby('series').mean()

    # Save the processed data
    data.to_csv(path1+'_mean_crps.csv',index=True)

    ax = data.plot(kind='line',
                   title='Average CRPS',
                   grid=True,
                   color=colors[:10],
                   legend=True,
                   figsize=(12, 9))
    ax.set_xlabel('Series Index')
    ax.set_ylabel('CRPS')
    plt.savefig(path+'_avg.png')
    #plt.show()
    plt.close()



if __name__=='__main__':

    # Get R results
    r_es_data = pd.read_csv(f'./Evaluation_Result_new/Energy_Score/Tourism.csv')
    r_crps_data = pd.read_csv(f'./Evaluation_Result_new/CRPS/Tourism.csv')
    r_vs_data = pd.read_csv(f'./Evaluation_Result_new/Variogram_Score/Tourism.csv')



    plot_avg_energy_score(r_es_data,
                          path1=f'./Analyze_Result_new/Energy_Score/Tourism',
                          path=f'./Plot_new/Tourism/Energy_Score')
    plot_avg_variogram_score(r_vs_data,
                             path1=f'./Analyze_Result_new/Variogram_Score/Tourism',
                             path=f'./Plot_new/Tourism/Variogram_Score')
    plot_avg_crps(r_crps_data,
                  path1=f'./Analyze_Result_new/CRPS/Tourism',
                  path=f'./Plot_new/Tourism/CRPS')