import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams.update({'font.size': 16})

# Read raw data
df = pd.read_csv('./Data/Simulated_Data_WithNoise.csv')
y = df.iloc[500,:].tolist()


# Read the params
with open('.\\Base_Forecasts\\WithNoise_ARIMA.json','r') as file1:
    params_arima = json.load(file1)

arima_params = []
for i in params_arima:
    arima_params.append(i[:2])
arima_params = np.array(arima_params)

with open('.\\Base_Forecasts\\WithNoise_ETS.json','r') as file2:
    params_ets = json.load(file2)

ets_params = []
for i in params_ets:
    ets_params.append(i[:2])
ets_params = np.array(ets_params)
arima_params = arima_params[0]
ets_params = ets_params[0]


# # ets
# axs[0].plot([i for i in range(169,229)],df_T[-60:], 'k-')
# axs[0].plot([i for i in range(217,229)], ets_params[:,0,0],'r-')
# axs[0].fill_between(x = [i for i in range(217,229)], y1 = ets_params[:,0,0]-2*np.sqrt(ets_params[:,1,0]), y2 = ets_params[:,0,0]+2*np.sqrt(ets_params[:,1,0]), alpha=0.5)
# axs[0].legend(["真实值", "预测均值", "2Sigma预测区间"], loc="upper left")
# axs[0].set_title('Tourism::ETS')
# axs[0].set_xlabel('Time')

# # deepar
# axs[1].plot([i for i in range(169,229)], df_T[-60:], 'k-')
# axs[1].plot([i for i in range(217,229)], deepar_params[:,0,0],'r-')
# axs[1].fill_between(x = [i for i in range(217,229)], y1 = deepar_params[:,0,0]-2*np.sqrt(deepar_params[:,1,0]), y2 = deepar_params[:,0,0]+2*np.sqrt(deepar_params[:,1,0]), alpha=0.5)
# axs[1].legend(["真实值", "预测均值", "2Sigma预测区间"], loc="upper left")
# axs[1].set_title('Tourism::DeepAR')
# axs[1].set_xlabel('Time')
fig,axs = plt.subplots(1,4,figsize = (12,9))
for i in range(4):
    axs[i].scatter([y[i],y[i]],[1,2])
    # axs[i].fill_betweenx(y=[1],x1=[y[i]-2*math.sqrt(arima_params[1][i])],x2=[y[i]+2*math.sqrt(arima_params[1][i])],color='blue',where=x2 >= x1,)
    # axs[i].fill_betweenx(y=[2],x1=[y[i]-2*math.sqrt(ets_params[1][i])],x2=[y[i]+2*math.sqrt(ets_params[1][i])],color='red')
    axs[i].errorbar([y[i]], [1], xerr=2*math.sqrt(arima_params[1][i]),ecolor='blue')
    axs[i].errorbar([y[i]], [2], xerr=2*math.sqrt(ets_params[1][i]),ecolor='red')


plt.tight_layout()

#plt.savefig('./Plot_new/Tourism/Base_Forecasts_tourism.png')
plt.show()