import numpy as np
import pandas as pd
import json
from numpy import random
from util import transform_fc

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
    term1 = np.sum(np.sum(np.square(dif1),axis=0),axis=0)
    term2 = np.sum(np.sum(np.square(dif2),axis=0),axis=0)
    return ((-0.5*term1)+term2)/(x.shape[1])

def variogram_score(y,x):
    '''
    Calculate the variogram score
    '''
    n = 111  # Variables num, equals to 7 when simulating
    Q = 1000  # Sample size
    y_list = []
    x_list = []
    for i in range(12):
        y_list.append(y[:,(i*Q):((i+1)*Q)])
        x_list.append(x[:,(i*Q):((i+1)*Q)])

    term1 = 0
    for k in range(12):
        for i in range(n-1):
            for j in range(i+1,n):
                term2 = 0
                for q in range(Q):
                    term2 = term2+abs(x_list[k][i,q]-x_list[k][j,q])
                term2 = term2/Q
                term1 = term1+(abs(y_list[k][i,0]-y_list[k][j,0])-term2)**2
    return term1


if __name__=='__main__':

    # Read transformed Smat and new_index
    with open('./Reconcile_and_Evaluation/Tourism_tranformed_Smat.json','r') as file:
        S = json.load(file)
    S = np.array(eval(S[0]))

    new_index = [i for i in range(1,111)]+[0]

    # Get data
    N = 216
    Q = 1000
    stride = 12

    data = pd.read_csv('./Data/Tourism/Tourism_process.csv').iloc[N:,2:].reset_index(drop=True)
    data_res = []
    for i in range(stride):
        data_res.append(np.array(data.iloc[i,:].tolist()))

    j = 0
    y = np.array([data_res[j]]*1000).T
    for j in range(1,stride):
        y = np.append(y,np.array([data_res[j]]*1000).T,axis=1)

    # Get base forecast
    # Process the base forecasts data
    with open('./Base_Forecasts/Tourism.json','r') as file2:
        fc = json.load(file2)
    
    
    fc_res = []
    fc_res.append(transform_fc(fc,stride))

    # with open('./Tourism_out_process.json','w') as f:
    #     f.write(json.dumps(fc_res))

    mean = []
    cov = []
    for k in range(stride):
        mean.append(np.array(fc_res[0][k][0]))
        cov.append(np.cov(fc_res[0][k][2]))

     # base forecasts samples
    i = 0
    x = np.random.multivariate_normal(mean[i], cov[i], Q).T
    xs = np.random.multivariate_normal(mean[i], cov[i], Q).T

    for i in range(1,stride):
        x = np.append(x,np.random.multivariate_normal(mean[i], cov[i], Q).T,axis=1)
        xs = np.append(xs,np.random.multivariate_normal(mean[i], cov[i], Q).T,axis=1)

    x1 = np.take(x, new_index, axis=0)
    x2 = np.take(xs, new_index, axis=0)
    y1 = np.take(y, new_index, axis=0)
    

    G = np.load(f'./Reconcile_and_Evaluation/Tourism_G_early_stop_2.npy')

    # save the 5 times results
    CRPS = []
    ES = []
    VS = []

    res1 = crps(y1,S@G@x1,S@G@x2)
    res_crps = res1.tolist()
    #print(res_crps)
    # Recover
    new_res_crps = []
    j=0
    for i in new_index:
        new_res_crps.append(res_crps[new_index.index(j)])
        j+=1
    #print(new_res_crps)
    CRPS.append(new_res_crps)

    res2 = energy_score(y1,S@G@x1,S@G@x2)
    res_energy_score = res2
    ES.append(res_energy_score)

    res3 = variogram_score(y1,S@G@x1)
    #res_variogram_score = [res3[m,0] for m in range(res3.shape[0])]
    VS.append(res3)


    dic={'CRPS':CRPS,'ES':ES,'VS':VS}
    with open(f'./Evaluation_Result/Results_Opt/Tourism.json', 'w') as file3:
        file3.write(json.dumps(dic))