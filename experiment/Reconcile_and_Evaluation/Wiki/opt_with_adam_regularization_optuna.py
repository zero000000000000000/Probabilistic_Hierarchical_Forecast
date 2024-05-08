from autograd import grad
import autograd.numpy as np
from tqdm import tqdm
import pandas as pd
import json
from util import transform_fc
from early_stop import EarlyStopping
import optuna


def loss_function(G):
    global S,x1,x2,y1,l1_lambda,l2_lambda
    dif1 = S@G@x1-S@G@x2
    dif2 = y1-S@G@x1
    term1 = np.sum(np.sum(np.square(dif1),axis=0),axis=0)
    term2 = np.sum(np.sum(np.square(dif2),axis=0),axis=0)
    res1 = ((-0.5*term1)+term2)/(x1.shape[1])

    l1_regularization = l1_lambda*np.sum(np.abs(G[:149,:]))
    l2_regularization = l2_lambda*np.sum(np.square(G[:149,:]))

    res = res1+l1_regularization+l2_regularization
    return res


def objective(trial):
    '''
    grid search
    '''
    global G,m1,n,l1_lambda,l2_lambda
    l1_lambda = trial.suggest_categorical("l1_lambda",[100,1000,10000])
    l2_lambda = trial.suggest_categorical("l2_lambda",[100,1000,10000])

    G = np.zeros((m1,n))
    #G = np.load('./Reconcile_and_Evaluation/tourism_G_early_stop.npy')
    G[(m1-1),(n-1)] = 1

    gradient_function = grad(loss_function)
    dG = gradient_function(G)

    dG[(m1-1),] = [0]*n
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    learning_rate = 0.001
    t = 0
    num_iterations = 100
    m = np.zeros_like(dG)
    v = np.zeros_like(dG)

    loss = []
    # patience = 10
    early_stopping = EarlyStopping(patience=100)

    for iteration in tqdm(range(num_iterations)):
        loss_i = loss_function(G)
        loss.append(loss_i)

        if early_stopping(loss_i):
            print(f"Early stopping triggered at epoch {iteration + 1}.")
            break
        t += 1
        
        m = beta1 * m + (1 - beta1) * dG
        v = beta2 * v + (1 - beta2) * (dG ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        G -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        dG = gradient_function(G)
        dG[(m1-1),] = [0]*n

    return loss[-1]


if __name__ == '__main__':

    # Set params
    N = 351
    Q = 1000
    N1 = 291
    W = 4
    n = 199
    m1 = 150
    stride = 15

    # Read transformed Smat and new_index
    with open('./Reconcile_and_Evaluation/Wiki/Wiki_tranformed_Smat.json','r') as file:
        S = json.load(file)
    S = np.array(eval(S[0]))

    new_index = [i for i in range(1,199)]+[0]

    # Process the base forecasts data
    with open('./Base_Forecasts/Wiki/Wiki_in.json','r') as file2:
        fc = json.load(file2)
    
    fc_list = []
    for i in range(W):
        fc_list.append(fc[i*n:(i+1)*n])
    
    fc_res = []
    for i in fc_list:
        fc_res.append(transform_fc(i,stride))

    mean = []
    cov = []
    for i in range(W):
        for k in range(stride):
            mean.append(np.array(fc_res[i][k][0]))
            cov.append(np.diag(np.array(fc_res[i][k][1])))

    # Read raw data
    data = pd.read_csv('./Data/Wiki/Wiki_process.csv').iloc[N1:N,1:].reset_index(drop=True)
    data_res = []
    for i in range(W*stride):
        data_res.append(np.array(data.iloc[i,:].tolist()))
    
    ####
    # base forecasts samples
    i = 0
    x = np.random.multivariate_normal(mean[i], cov[i], Q).T
    xs = np.random.multivariate_normal(mean[i], cov[i], Q).T

    for i in range(1,W*stride):
        x = np.append(x,np.random.multivariate_normal(mean[i], cov[i], Q).T,axis=1)
        xs = np.append(xs,np.random.multivariate_normal(mean[i], cov[i], Q).T,axis=1)

    # raw data samples
    j = 0
    y = np.array([data_res[j]]*1000).T
    for j in range(1,W*stride):
        y = np.append(y,np.array([data_res[j]]*1000).T,axis=1)

    x1 = np.take(x, new_index, axis=0)
    x2 = np.take(xs, new_index, axis=0)
    y1 = np.take(y, new_index, axis=0)

    x1 = x1[:,-15000:]
    x2 = x2[:,-15000:]
    y1 = y1[:,-15000:]

    search_space = {'l1_lambda':[100,1000,10000],
                    'l2_lambda':[100,1000,10000]}

    study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space,seed=123),
    direction = "minimize",
    study_name = "grid_task")

    study.optimize(objective)

    # 3, 获取最优超参
    best_params = study.best_params
    best_value = study.best_value
    print("best_value = ",best_value)
    print("best_params:",best_params)
    with open('./Reconcile_and_Evaluation/Wiki/regularization_params.json','w') as file:
        file.write(json.dumps(best_params))