import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

colors = ['blue','green','red','cyan','magenta','yellow','black','orange','purple','brown','pink']

parser = argparse.ArgumentParser()
parser.add_argument('--metrics', type=str, default='Energy_Score', help='Evaluation metrics')
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')
parser.add_argument('--basefdep', type=str, default='Independent', help='The base forecast independence')

def plot_energy_score(data,path):
    '''
    Plot the line of energy score with different reconciliation method and time step
    '''
    global colors
    data = data.iloc[:,1:]     
    ax = data.plot(kind='line',
                   x='t',
                   title='Energy Score For Time Step',
                   grid=True,
                   color=colors,
                   legend=True,
                   figsize=(12, 9))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Energy Score')
    ax.set_xticks([i for i in range(1,len(data)+1)])
    ax.set_xticklabels([i for i in range(1,len(data)+1)])
    plt.savefig(path+'.png')
    #plt.show()
    plt.close()

def plot_avg_energy_score(data,path):
    '''
    Plot the line of average energy score with different reconciliation method
    '''
    global colors
    data = data.iloc[:,1:(len(data.columns)-1)]
    data = data.mean()
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
    plt.savefig(path+'_avg.png')
    #plt.show()
    plt.close()

def plot_avg_crps(data,path):
    '''
    Plot the line of average crps with different reconciliation method and time step
    '''
    global colors
    data = data.iloc[:,1:(len(data.columns)-1)]
    data = data.groupby('series').mean()
    # Save the processed data
    data.to_csv(path+'_avg.csv',index=True)

    ax = data.plot(kind='line',
                   title='Average CRPS',
                   grid=True,
                   color=colors,
                   legend=True,
                   figsize=(12, 9))
    ax.set_xlabel('Series Index')
    ax.set_ylabel('CRPS')
    plt.savefig(path+'_avg.png')
    #plt.show()
    plt.close()

if __name__ == '__main__':
    # Get args
    args = parser.parse_args()
    metrics = args.metrics
    generate = args.generate
    rootbasef = args.rootbasef
    basefdep = args.basefdep

    # Create path
    save_path = f'./Plot/{metrics}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' has been created.")

    path = save_path + f'/{generate}_{rootbasef}_{basefdep}'

    # Read the dataset
    data = pd.read_csv(f'./Evaluation_Result/{metrics}/{generate}_{rootbasef}_{basefdep}.csv')

    if metrics == 'Energy_Score':
        plot_energy_score(data,path)
        plot_avg_energy_score(data,path)
    else:
        #plot_crps(data,path)
        plot_avg_crps(data,path)