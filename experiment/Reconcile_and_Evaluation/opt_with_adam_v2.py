from autograd import grad
import autograd.numpy as np
from tqdm import tqdm
import pandas as pd
import json
from util import transform_fc
import matplotlib.pyplot as plt
from early_stop import EarlyStopping

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def loss_function(G):
    global S,x1,x2,y1
    dif1 = S@G@x1-S@G@x2
    dif2 = y1-S@G@x1
    term1 = np.sum(np.sum(np.square(dif1),axis=0),axis=0)
    term2 = np.sum(np.sum(np.square(dif2),axis=0),axis=0)

    return ((-0.5*term1)+term2)/(x1.shape[1])

# def early_stopping(loss, patience, min_delta=100):
#     '''
#     whether to early stop
#     '''
#     if len(loss) < patience:
#         return False
#     else:
#         count = 1
#         best_loss = min(loss)
#         for i in range(loss.index(best_loss),len(loss)):
#             if (abs(loss[i] -best_loss) < min_delta) | (loss[i]-best_loss >= min_delta):
#                 count+=1
#         if count > patience:
#             return True
#     return False

if __name__ == '__main__':


    # Set params
    N = 216
    Q = 1000
    N1 = 168
    W = 4
    n = 111
    m1 = 76
    stride = 12

    # Read transformed Smat and new_index
    with open('./Reconcile_and_Evaluation/Tourism_tranformed_Smat.json','r') as file:
        S = json.load(file)
    S = np.array(eval(S[0]))

    new_index = [i for i in range(1,111)]+[0]

    # Process the base forecasts data
    with open('./Base_Forecasts/Tourism_in.json','r') as file2:
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
            cov.append(np.cov(fc_res[i][k][2]))

    # Read raw data
    data = pd.read_csv('./Data/Tourism/Tourism_process.csv').iloc[N1:N,2:].reset_index(drop=True)
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

    # G = np.zeros((m1,n))
    G = np.load('./Reconcile_and_Evaluation/tourism_G_early_stop.npy')
    G[(m1-1),(n-1)] = 1

    
    gradient_function = grad(loss_function)
    dG = gradient_function(G)

    dG[(m1-1),] = [0]*n
    
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-7
    learning_rate = 0.001
    t = 0
    num_iterations = 10000
    m = np.zeros_like(dG)
    v = np.zeros_like(dG)

    loss = []
    # patience = 10
    early_stopping = EarlyStopping(patience=400)

    for iteration in tqdm(range(num_iterations)):
        loss_i = loss_function(G)
        loss.append(loss_i)

        if early_stopping(loss_i):
            print(f"Early stopping triggered at epoch {num_iterations + 1}.")
            break
        t += 1
        
        m = beta1 * m + (1 - beta1) * dG
        v = beta2 * v + (1 - beta2) * (dG ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        G -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        dG = gradient_function(G)
        dG[(m1-1),] = [0]*n

        

    np.save('./Reconcile_and_Evaluation/tourism_G_early_stop_2.npy',G)

    x_axis = list(range(1,len(loss)+1))
    plt.plot(x_axis,loss,'-r')
    plt.title('能量得分迭代图')
    plt.xlabel('迭代次数')
    plt.ylabel('能量得分')
    plt.savefig('./Tourism_GD_early_stop_2.png')
    plt.show()


