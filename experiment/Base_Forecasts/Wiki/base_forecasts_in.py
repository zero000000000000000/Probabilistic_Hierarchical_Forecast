import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
import json
import time
import multiprocessing
from tqdm import tqdm
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Wiki', help='Type of dataset')

def Create_Data_Window(df,N,W,stride):
    '''
    Create W windows
    '''
    lis = []
    for i in range(W):
        lis.append(df.iloc[:(N+i*stride),:].reset_index().iloc[:,1:])
    return lis

def ARIMA_Prob_Forecast(series,h):
    '''
    Forecast h step using Auto_ARIMA method
    '''
    global num
    forecaster = AutoARIMA(sp=1,start_p=1,start_q=0, max_p=3, max_q=3,suppress_warnings=True) 
    forecaster.fit(series, fh=range(1,h+1))
    var_pred = forecaster.predict_var()
    var_pred = var_pred.iloc[:,0].tolist()
    mu_pred = forecaster.predict()
    mu_pred = mu_pred.tolist()
    resid = forecaster.predict_residuals().fillna(0).tolist()
    fitted = (resid + series).tolist()
    print('{}/199'.format(num))
    num += 1
    return [mu_pred,var_pred,resid,fitted]


# Get args
# args = parser.parse_args()
# dataset = args.dataset
dataset = 'Wiki'
st = time.time()
# Set the params
N = 351 # train size
h = 15   # forecast towards 12
N1 = 291
stride = 15
W = 4

# Load the raw data
data = pd.read_csv(f'./Data/{dataset}/{dataset}_process.csv').iloc[:N,:]

# For every variable, forecasting h = 15
# The first base forecast of validation set
forecast_results_in = []

# Create windows
data_windows = Create_Data_Window(data,N1,W,stride)
#print([df['T'][len(df)-1] for df in data_windows])

# pool = multiprocessing.Pool(processes=4) # 创建4个进程
# results = []
num = 1
print('Start.....')
for j in range(199):
    forecast_results_in.append(ARIMA_Prob_Forecast(data_windows[0].iloc[:,(j+1)],15))
# pool.close()
# pool.join()
# print ("Sub-process(es) done.")

# for res in results:
#     forecast_results_in.append(res.get())
num = 1
print('Start.....')
for j in range(199):
    forecast_results_in.append(ARIMA_Prob_Forecast(data_windows[1].iloc[:,(j+1)],15))

num = 1
print('Start.....')
for j in range(199):
    forecast_results_in.append(ARIMA_Prob_Forecast(data_windows[2].iloc[:,(j+1)],15))

num = 1
print('Start.....')
for j in range(199):
    forecast_results_in.append(ARIMA_Prob_Forecast(data_windows[3].iloc[:,(j+1)],15))

res = []
for i in forecast_results_in:
    mu_pred = i[0].tolist()
    var_pred = i[1].iloc[:,0].tolist()
    resid = i[2]
    fitted = i[3]
    res.append([mu_pred,var_pred,resid,fitted])

# Save the results
with open(f'./Base_Forecasts/{dataset}/{dataset}_in.json','w') as file:
    file.write(json.dumps(forecast_results_in))

et = time.time()
print(et-st)