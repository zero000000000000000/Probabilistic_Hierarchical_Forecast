import numpy as np
import pandas as pd
from numpy import linalg as LA
import gurobipy as gp
import argparse
import json
from gurobipy import GRB
from numpy import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')
parser.add_argument('--basefdep', type=str, default='Independent', help='The base forecast independence')

def squa(y):
    '''
    Calculate the square of the matrix
    '''
    s=0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            s+=y[i,j]*y[i,j]
    return s

if __name__=='__main__':

    st = time.time()

    # Get params
    args = parser.parse_args()
    generate = args.generate
    rootbasef = args.rootbasef
    basefdep = args.basefdep

    # Get raw data
    N1 = 490    # Validation set start
    N = 500     # Training set start
    Q = 1000    # Sample size
    W = 10      # Windows

    data = pd.read_csv(f'./Data/Simulated_Data_{generate}.csv').iloc[N1:N,:]
    
    y1 = np.array(data.iloc[0,:].tolist())
    y = np.matrix([y1]*1000).T
    for i in range(1,len(data)):
        y1 = np.array(data.iloc[i,:].tolist())
        y = np.append(y,np.matrix([y1]*1000).T,axis=1)

    # Get base forecast
    with open(f'./Base_Forecasts/{generate}_{rootbasef}_in.json') as file:
        fc = json.load(file)
    i=0
    mean = fc[i][0]
    var = fc[i][1]
    cov = np.diag(var)
    x = random.multivariate_normal(mean, cov, Q).T
    xs = random.multivariate_normal(mean, cov, Q).T

    for i in range(1,W):
        mean = fc[i][0]
        var = fc[i][1]
        cov = np.diag(var)
        x = np.append(x,random.multivariate_normal(mean, cov, Q).T,axis=1)
        xs = np.append(xs,random.multivariate_normal(mean, cov, Q).T,axis=1)


    # Get S and index and new_index
    S = np.matrix(np.array([[0,-1,-1,1],[0,1,1,0],[-1,-1,-1,1],[1,0,0,0],
                            [0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    new_index = [1,2,3,4,5,6,0]
    # Permute
    x1 = np.take(x, new_index, axis=0)
    x2 = np.take(xs, new_index, axis=0)
    y = np.take(y, new_index, axis=0)

    
    # Get validation set
    x1 = x1[:,-4000:]
    x2 = x2[:,-4000:]
    y = y[:,-4000:]

    # Create env and model
    print('Start to search the solution....')

    env = gp.Env(empty=True)
    env.start()
    model = gp.Model('ALL', env=env) # the optimization model

    # Get G
    p = 3
    q = 7

    G = []

    for i in range(p):
        row = []
        for j in range(q):
            #var = model.addVar(ub=1, lb=-1, vtype=GRB.CONTINUOUS, name=f"G_({i}, {j})")
            var = model.addVar(vtype=GRB.CONTINUOUS, name=f"G_({i}, {j})")
            row.append(var)
        G.append(row)
    G1 = np.matrix(G)
    G1 = np.append(G1,[[0,0,0,0,0,0,1]],axis=0)

    # I = np.eye(4)
    # model.addConstr(I.reshape(-1) == np.kron(S.T, I) @ G1, "c_1")

    part1 = squa(0.5 * (S @ G1 @ x1 - S @ G1 @ x2))
    part2 = squa((y - S @ G1 @ x1))
    model.setObjective((-part1 + part2)/4000, GRB.MINIMIZE)
    model.params.TimeLimit = 120
    model.optimize()

    res = np.zeros((28,1))

    print(f"Optimal value: {model.objVal}")
    i = 0
    for v in model.getVars():
        print(f"{v.varName}: {v.x}")
        res[i] = v.x
        i+=1
    res = res.reshape((4,7))
    res[3,6] = 1
    
    np.save(f'./Reconcile_and_Evaluation/Gurobipy_Results/{generate}_{rootbasef}_{basefdep}_Gopt.npy',res)

    et = time.time()
    print(et-st)