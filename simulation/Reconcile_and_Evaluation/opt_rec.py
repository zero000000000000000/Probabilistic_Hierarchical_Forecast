import numpy as np
import pandas as pd
import argparse
import json
from numpy import random

parser = argparse.ArgumentParser()
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')
parser.add_argument('--basefdep', type=str, default='Independent', help='The base forecast independence')


def crps(y,x,xs):
    '''
    Calculate the CRPS
    '''
    dif1 = x-xs
    dif2 = y-x
    term1 = np.sum(np.abs(dif1),axis=1)
    term2 = np.sum(np.abs(dif2),axis=1)
    return ((-0.5*term1)+term2)/(x.shape[1])

def energy_score(y,x,xs):
    '''
    Calculate the energy score
    '''
    dif1 = x-xs
    dif2 = y-x
    term1 = np.sum(np.sum(np.square(dif1),axis=0),axis=1)
    term2 = np.sum(np.sum(np.square(dif2),axis=0),axis=1)
    return ((-0.5*term1)+term2)/(x.shape[1])

if __name__=='__main__':
    # Get params
    args = parser.parse_args()
    generate = args.generate
    rootbasef = args.rootbasef
    basefdep = args.basefdep
    immutable = args.immutable

    # Get data
    N = 500
    Q = 1000
    data = pd.read_csv(f'./Data/Simulated_Data_{generate}.csv').iloc[N,:]
    y1 = np.array(data.tolist())
    y = np.matrix([y1]*1000).T

    # Get base forecast
    with open(f'./Base_Forecasts/{generate}_{rootbasef}.json') as file:
        fc = json.load(file)
    mean = fc[0][0]
    var = fc[0][1]
    cov = np.diag(var)

    # Get S and index and new_index
    S = np.matrix(np.array([[0,-1,-1,1],[0,1,1,0],[-1,-1,-1,1],[1,0,0,0],
                            [0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    new_index = [1,2,3,4,5,6,0]

    G = np.load(f'./Reconcile_and_Evaluation/gurobipy_results/{generate}_{rootbasef}_{basefdep}_Gopt.npy')

    # save the 5 times results
    CRPS = []
    ES = []
    for i in range(5):
        # Get x and xs
        x = random.multivariate_normal(mean, cov, Q).T
        xs = random.multivariate_normal(mean, cov, Q).T

        # Permute
        if immutable == 'immutable':
            x1 = np.take(x, new_index, axis=0)
            x2 = np.take(xs, new_index, axis=0)
            y1 = np.take(y, new_index, axis=0)
        else:
            x1 = x
            x2 = xs
            y1 = y

        res1 = crps(y1,S@G@x1,S@G@x2)
        res_crps = [res1[m,0] for m in range(res1.shape[0])]
        CRPS.append(res_crps)

        res2 = energy_score(y1,S@G@x1,S@G@x2)
        res_energy_score = [res2[m,0] for m in range(res2.shape[0])]
        ES.append(res_energy_score)


    dic={'CRPS':CRPS,'ES':ES}
    with open(f'./Evaluation_Result/Results_Opt/{generate}_{rootbasef}_{basefdep}.json', 'w') as file:
        file.write(json.dumps(dic))