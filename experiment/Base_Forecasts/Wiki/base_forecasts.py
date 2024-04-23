import pandas as pd
import numpy as np
from sktime.forecasting.arima import AutoARIMA
import json
import time
import multiprocessing
from tqdm import tqdm


# def ETS_Prob_Forecast(series,h,sp):
#     '''
#     Forecast h step using ETS method without seasonal component
#     '''
#     model = ExponentialSmoothing(endog=series, trend=True,seasonal=sp)
#     es_fit_result = model.fit()
#     res_df = es_fit_result.get_forecast(h).summary_frame()
#     var_pred = (res_df['mean_se']**2).tolist()
#     mu_pred = (res_df['mean']).tolist()
#     fitted = es_fit_result.fittedvalues.tolist()
#     resid = (fitted - series).tolist()
#     return [mu_pred,var_pred,resid,fitted]

def ARIMA_Prob_Forecast(series,h):
    '''
    Forecast h step using Auto_ARIMA method
    '''
    forecaster = AutoARIMA(sp=1,start_p=1,start_q=0,max_p=3, max_q=3,suppress_warnings=True) 
    forecaster.fit(series, fh=range(1,h+1))
    var_pred = forecaster.predict_var()
    var_pred = var_pred.iloc[:,0].tolist()
    mu_pred = forecaster.predict()
    mu_pred = mu_pred.tolist()
    resid = forecaster.predict_residuals().fillna(0).tolist()
    fitted = (resid + series).tolist()
    return [mu_pred,var_pred,resid,fitted]

if __name__ == '__main__':

    st = time.time()
    # Set the params
    dataset = 'Wiki'
    N = 351 # train size
    h = 15   # forecast towards 12
    sp = 1

    # Load the raw data
    data = pd.read_csv(f'./Data/{dataset}/{dataset}_process.csv').iloc[:N,:]
    forecast_results = []

    pool = multiprocessing.Pool(processes=4) # 创建4个进程
    results = []
    for i in tqdm(range(199)):
        results.append(pool.apply_async(ARIMA_Prob_Forecast, (data.iloc[:,(i+1)],15, )))
    pool.close()
    pool.join()
    print ("Sub-process(es) done.")
  
    for res in results:
        forecast_results.append(res.get())
    
    res = []
    for i in forecast_results:
        mu_pred = i[0].tolist()
        var_pred = i[1].iloc[:,0].tolist()
        resid = i[2]
        fitted = i[3]
        res.append([mu_pred,var_pred,resid,fitted])

    # Save the results
    with open(f'./Base_Forecasts/{dataset}/{dataset}.json','w') as file:
        file.write(json.dumps(res))

    et = time.time()
    print(et-st)