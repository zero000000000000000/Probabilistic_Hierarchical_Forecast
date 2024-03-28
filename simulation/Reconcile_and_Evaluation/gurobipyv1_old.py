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
    N = 500
    Q = 1000
    data = pd.read_csv(f'./Data/Simulated_Data_{generate}.csv').iloc[(N-1),:]
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

    # Get x and xs
    x = random.multivariate_normal(mean, cov, Q).T
    xs = random.multivariate_normal(mean, cov, Q).T

    # Permute
    x1 = np.take(x, new_index, axis=0)
    x2 = np.take(xs, new_index, axis=0)
    y = np.take(y, new_index, axis=0)


    # Create env and model
    OutputFlag=0
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",OutputFlag)
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
            var = model.addVar(lb=0,vtype=GRB.CONTINUOUS, name=f"G_({i}, {j})")
            row.append(var)
        G.append(row)
    G1 = np.matrix(G)
    G1 = np.append(G1,[[0,0,0,0,0,0,1]],axis=0)

    I = np.eye(4)
    part = G1@S
    model.addConstr(part==I, "c_1")

    part1 = squa(0.5 * (S @ G1 @ x1 - S @ G1 @ x2))
    part2 = squa((y - S @ G1 @ x1))
    model.setObjective((-part1 + part2)/1000, GRB.MINIMIZE)
    model.optimize()

    res = np.zeros((28,1))
    if model.status == GRB.Status.OPTIMAL:
        print(f"Optimal value: {model.objVal}")
        i = 0
        for v in model.getVars():
            print(f"{v.varName}: {v.x}")
            res[i] = v.x
            i+=1
        res = res.reshape((4,7))
        res[3,6] = 1
    else:
        print("No optimal solution found.")
    
    print(G)
    np.save('./G_4.npy',res)
    et = time.time()
    print(et-st)