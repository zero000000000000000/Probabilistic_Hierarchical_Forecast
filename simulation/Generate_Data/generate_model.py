import numpy as np
import pandas as pd
import os

def Check_poly(lis,category):
    '''
    Check the stationary and invertible parameter space
    '''
    if category == 'AR':
        lis = [-1*x for x in lis][::-1]+[1]
    else:
        lis = [x for x in lis][::-1]+[1]
    print(lis)
    if np.all(np.abs(np.roots(lis))>1):
        print(np.abs(np.roots(lis)))
        return True
    else:
        return False

def Generate_params(ins,category):
    '''
    Generate the coefficient of AR and MA
    '''
    params = []
    if ins > 0:
        params_first = round(np.random.uniform(0.5,0.7),2)
        if ins == 2:
            if category == 'AR':
                params_real = round(np.random.uniform(params_first-0.9,0.9-params_first),2)
            else:
                params_real = round(np.random.uniform(-(params_first+0.9)/3.2,(0.9+params_first)/3.2),2)
            params = [params_real,params_first]
        else:
            params = [params_first]
        print(Check_poly(params,category))
        return params
    return []

# Set constants
m = 4   # number of bottom series
N = 2000 # size of sample

# Generate the contemporaneous error with a multivariate normal distribution
#bottom_err_cov = np.array([[5,3,2,1],[3,4,2,1],[2,2,5,3],[1,1,3,4]])
#bottom_err = np.random.multivariate_normal(np.array([0,0,0,0]),bottom_err_cov,N)

# Generate orders and params of the ARIMA model
# Set seed
np.random.seed(12)
p_order = np.random.randint(0,3,size=m)
d_order = np.random.randint(0,2,size=m)
q_order = np.random.randint(0,3,size=m)

ar_params, ma_params = [],[]

for i in range(m):
    ar_params.append(Generate_params(p_order[i],'AR'))
    ma_params.append(Generate_params(q_order[i],'MA'))

# Save the 8 ARIMA models
dic = {'bottom_series_index':[1,2,3,4],'p':p_order.tolist(),'d':d_order.tolist(),'q':q_order.tolist(),'phi':ar_params,'theta':ma_params}
df = pd.DataFrame(dic)
#print(os.getcwd())
#os.chdir('./simulation')
print(os.getcwd())
os.makedirs('Data')
df.to_csv('./Data/Generate_ARIMA_Model.csv',index=False)