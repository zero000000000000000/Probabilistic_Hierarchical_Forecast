import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')

def Create_Data_Window(df,N,W):
    '''
    Create W windows
    '''
    lis = []
    for i in range(W):
        lis.append(df.iloc[:(N+i),:].reset_index().iloc[:,1:])
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

def ETS_Prob_Forecast(series,h):
    '''
    Forecast h step using ETS method without seasonal component
    '''
    model = ExponentialSmoothing(endog=series, trend=True)
    es_fit_result = model.fit()
    res_df = es_fit_result.get_forecast(h).summary_frame()
    var_pred = res_df.iloc[0,1]**2
    mu_pred = res_df.iloc[0,0]
    fitted = es_fit_result.fittedvalues.tolist()
    resid = (fitted - series).tolist()
    return [mu_pred,var_pred,resid,fitted]

def Forecast_Base(df,approach,h):
    '''
    Create base forecast
    '''
    series_mu, series_var, series_resid, series_fitted= [], [], [], []
    for i in range(len(df.columns)):
        if (i==0) & (approach == 'ETS'):
            [mu_pred,var_pred,resid,fitted] = ETS_Prob_Forecast(df.iloc[:,i],h)
            series_mu.append(mu_pred)
            series_var.append(var_pred)
            series_resid.append(resid)
            series_fitted.append(fitted)
        else:
            [mu_pred,var_pred,resid,fitted] = ARIMA_Prob_Forecast(df.iloc[:,i],h)
            series_mu.append(mu_pred)
            series_var.append(var_pred)
            series_resid.append(resid)
            series_fitted.append(fitted)

    return [series_mu, series_var, series_resid,series_fitted]

if __name__=='__main__':
    # Get args
    args = parser.parse_args()
    generate = args.generate
    rootbasef = args.rootbasef

    # Set the params
    N = 500 # train size
    h = 1   # forecast towards 1
    N1 = 490    # Validation set start
    W = 10  # 10 expanding windows


    # Load the original data
    data = pd.read_csv(f'./Data/Simulated_Data_{generate}.csv').iloc[:(N+1),:]

    # For every variable, forecasting h = 1
    # The first base forecast of validation set
    forecast_results_in = []

    # Create windows
    data_windows = Create_Data_Window(data,N1,W)

    for i in tqdm(range(W)):
        forecast_results_in.append(Forecast_Base(data_windows[i],rootbasef,h))

    # The second base forecast of test set at 501
    forecast_results = []
    forecast_results.append(Forecast_Base(data.iloc[:N,],rootbasef,h))

    # Load the result in a json file
    with open(f'./Base_Forecasts/{generate}_{rootbasef}_in.json', 'w') as json_file:
        json.dump(forecast_results_in, json_file)

    with open(f'./Base_Forecasts/{generate}_{rootbasef}.json', 'w') as json_file:
        json.dump(forecast_results, json_file)