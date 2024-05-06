from autograd import grad
import autograd.numpy as np
from tqdm import tqdm
import pandas as pd
import json
generate = 'WithNoise'
rootbasef = 'ARIMA'
basefdep = 'Independent'
# Get data
N = 500
Q = 1000
N1 = 496
W = 4
data = pd.read_csv(f'../simulation/Data/Simulated_Data_{generate}.csv').iloc[N1:N,:]
y1 = np.array(data.iloc[0,:].tolist())
y = np.matrix([y1]*1000).T
for i in range(1,len(data)):
    y1 = np.array(data.iloc[i,:].tolist())
    y = np.append(y,np.matrix([y1]*1000).T,axis=1)

# Get base forecast
with open(f'../simulation/Base_Forecasts/{generate}_{rootbasef}_in.json') as file:
    fc = json.load(file)

i = 0
mean = fc[i][0]
if basefdep == 'Independent':
    var = fc[i][1]
    cov = np.diag(var)
else:
    cov = np.cov(fc[i][2])
x = np.random.multivariate_normal(mean, cov, Q).T
xs = np.random.multivariate_normal(mean, cov, Q).T

for i in range(1,W):
    mean = fc[i][0]
    if basefdep == 'Independent':
        var = fc[i][1]
        cov = np.diag(var)
    else:
        cov = np.cov(fc[i][2])
    x = np.append(x,np.random.multivariate_normal(mean, cov, Q).T,axis=1)
    xs = np.append(xs,np.random.multivariate_normal(mean, cov, Q).T,axis=1)

S = np.array([[0,-1,-1,1],[0,1,1,0],[-1,-1,-1,1],[1,0,0,0],
                        [0,1,0,0],[0,0,1,0],[0,0,0,1]])
new_index = [1,2,3,4,5,6,0]

x1 = np.take(x, new_index, axis=0)
x2 = np.take(xs, new_index, axis=0)
y = np.take(y, new_index, axis=0)


def loss_function(G):
    global S,y,x1,x2
    dif1 = S@G@x1-S@G@x2
    dif2 = y-S@G@x1
    term1 = np.sum(np.sum(np.square(dif1),axis=0),axis=0)
    term2 = np.sum(np.sum(np.square(dif2),axis=0),axis=0)
    return ((-0.5*term1)+term2)/(x1.shape[1])


#G0 = np.load('d:\HierarchicalCode\simulation\Reconcile_and_Evaluation\Gurobipy_Results\WithNoise_ARIMA_Independent_Gopt.npy')
G0 = np.zeros((4,7))
G0[3,6] = 1
gradient_function = grad(loss_function)
dG = gradient_function(G0)

dG[3,] = [0]*7
#G0 = G0.flatten()
#dG = dG.flatten()

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.001
t = 0
num_iterations = 1000
m = np.zeros_like(dG)
v = np.zeros_like(dG)

for iteration in tqdm(range(num_iterations)):
    t += 1
    
    m = beta1 * m + (1 - beta1) * dG
    v = beta2 * v + (1 - beta2) * (dG ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    G0 -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    dG = gradient_function(G0)
    dG[3,] = [0]*7
    #dG = dG.flatten()
np.save('./3.npy',G0)