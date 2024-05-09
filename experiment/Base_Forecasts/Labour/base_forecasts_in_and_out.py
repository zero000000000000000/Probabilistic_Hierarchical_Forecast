import pandas as pd
import numpy as np
# from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA
import json
import time
import multiprocessing
from tqdm import tqdm
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Tourism', help='Type of dataset')

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
    forecaster = AutoARIMA(sp=1,start_p=1,start_q=0, max_p=3, max_q=3,suppress_warnings=True) 
    forecaster.fit(series, fh=range(1,h+1))
    var_pred = forecaster.predict_var()
    var_pred = var_pred.iloc[:,0].tolist()
    mu_pred = forecaster.predict()
    mu_pred = mu_pred.tolist()
    resid = forecaster.predict_residuals().fillna(0).tolist()
    fitted = (resid + series).tolist()
    return [mu_pred,var_pred,resid,fitted]

if __name__ == '__main__':

    # Get args
    # args = parser.parse_args()
    # dataset = args.dataset
    dataset = 'labour'
    st = time.time()
    # Set the params
    N = 157 # train size
    h = 6   # forecast towards 12
    N1 = 133
    stride = 6
    W = 4

    # Load the raw data
    data = pd.read_csv(f'./Data/{dataset}/{dataset}_process_place.csv').iloc[:N,:]

    # For every variable, forecasting h = 6
    # The first base forecast of validation set
    forecast_results_in = []

    # Create windows
    data_windows = Create_Data_Window(data,N1,W,stride)
    # print([df['T'][len(df)-1] for df in data_windows])

    # pool = multiprocessing.Pool(processes=4) # 创建4个进程
    # results = []
    k = 1
    for i in range(4):
        for j in range(57):
            forecast_results_in.append(ARIMA_Prob_Forecast(data_windows[i].iloc[:,j],6, ))
            print(f'finish {k}/228')
            k += 1

    # data1 = pd.read_csv(f'./Data/{dataset}/{dataset}_process_place.csv').iloc[:N,:]
    forecast_results = []
    k1 = 1
    for j in range(57):
        forecast_results.append(ARIMA_Prob_Forecast(data.iloc[:,j],6, ))
        print(f'finish {k1}/228')
        k1 += 1
    # Save the results
    with open(f'./Base_Forecasts/{dataset}/{dataset}_in.json','w') as file:
        file.write(json.dumps(forecast_results_in))

    et = time.time()
    print(et-st)