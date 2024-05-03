import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
import json
import time
import multiprocessing
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Tourism', help='Type of dataset')

def ETS_Prob_Forecast(series,h,sp):
    '''
    Forecast h step using ETS method without seasonal component
    '''
    model = ExponentialSmoothing(endog=series, trend=True,seasonal=sp)
    es_fit_result = model.fit()
    res_df = es_fit_result.get_forecast(h).summary_frame()
    var_pred = (res_df['mean_se']**2).tolist()
    mu_pred = (res_df['mean']).tolist()
    fitted = es_fit_result.fittedvalues.tolist()
    resid = (fitted - series).tolist()
    return [mu_pred,var_pred,resid,fitted]

if __name__ == '__main__':

    # Get args
    args = parser.parse_args()
    dataset = args.dataset

    st = time.time()
    # Set the params
    N = 216 # train size
    h = 12   # forecast towards 12
    sp = 12

    # Load the raw data
    data = pd.read_csv(f'./Data/{dataset}/{dataset}_process.csv').iloc[:N,:]
    forecast_results = []

    pool = multiprocessing.Pool(processes=4) # 创建4个进程
    results = []
    for i in tqdm(range(111)):
        results.append(pool.apply_async(ETS_Prob_Forecast, (data.iloc[:,(i+2)],12,12, )))
    pool.close()
    pool.join()
    print ("Sub-process(es) done.")
  
    for res in results:
        forecast_results.append(res.get())
    
    # Save the results
    with open(f'./Base_Forecasts/{dataset}/{dataset}.json','w') as file:
        file.write(json.dumps(forecast_results))

    et = time.time()
    print(et-st)