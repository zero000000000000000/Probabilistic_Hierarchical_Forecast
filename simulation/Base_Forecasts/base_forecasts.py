import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
from tqdm import tqdm
import json

def Create_Data_Window(df,N,W):
    '''
    Create W windows
    '''
    lis = []
    for i in range(W):
        lis.append(df.iloc[i:(N+i),:].reset_index().iloc[:,1:])
    return lis

def ARIMA_Prob_Forecast(series,h):
    '''
    Forecast h step using Auto_ARIMA method
    '''
    forecaster = AutoARIMA(sp=1,start_p=1,start_q=0,max_p=3, max_q=3,suppress_warnings=True) 
    forecaster.fit(series, fh=h)
    var_pred = forecaster.predict_var()
    #print(var_pred)
    var_pred = var_pred.iloc[0,0]
    #print(var_pred)
    mu_pred = forecaster.predict()
    mu_pred = mu_pred.iloc[0]
    resid = forecaster.predict_residuals().fillna(0).tolist()
    #print(resid)
    fitted = (resid + series).tolist()
    return [mu_pred,var_pred,resid,fitted]

def Forecast_Base(df,approach):
    '''
    Create base forecast
    '''
    series_mu, series_var, series_resid, series_fitted= [], [], [], []
    for i in range(len(df.columns)):
        if approach == 'arima':
            [mu_pred,var_pred,resid,fitted] = ARIMA_Prob_Forecast(df.iloc[:,i],1)
            series_mu.append(mu_pred)
            series_var.append(var_pred)
            series_resid.append(resid)
            series_fitted.append(fitted)
    return [series_mu, series_var, series_resid,series_fitted]

# Load the original data
data = pd.read_csv('./Data/Simulated_Data_WithNoise.csv').iloc[:510,:]

# Set the params
N = 500
W = 10

# Create Windows
data_windows = Create_Data_Window(data,N,W)

# For every window and variable, forecasting h = 1
forecast_results = []

for i in tqdm(range(W)):
    forecast_results.append(Forecast_Base(data_windows[i],'arima'))

# Load the result in a json file
with open('WithNoise_ARIMA.json', 'w') as json_file:
    json.dump(forecast_results, json_file)