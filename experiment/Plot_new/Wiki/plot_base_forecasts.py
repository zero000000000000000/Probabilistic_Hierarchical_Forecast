import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# Read raw data
df = pd.read_csv('./Data/Wiki/Wiki_process.csv')
df_T = df['Total'].tolist()

# Read the params
with open('./Reconcile_and_Evaluation/Wiki/Wiki_out_process.json','r') as file1:
    params_arima = json.load(file1)

arima_params = []
for i in params_arima:
    arima_params.append(i[:2])
ets_params = np.array(arima_params)

with open('./Base_Forecasts/Wiki/Wiki_deepar_e100.json','r') as file2:
    params_deepar = json.load(file2)

deepar_params = np.array(params_deepar)

fig,axs = plt.subplots(2,1,figsize = (12,9))

# ets
axs[0].plot([i for i in range(292,367)],df_T[-75:], 'k-')
axs[0].plot([i for i in range(352,367)], ets_params[:,0,0],'r-')
axs[0].fill_between(x = [i for i in range(352,367)], y1 = ets_params[:,0,0]-2*np.sqrt(ets_params[:,1,0]), y2 = ets_params[:,0,0]+2*np.sqrt(ets_params[:,1,0]), alpha=0.5)
axs[0].legend(["True", "Mean", "2_Sigma"], loc="upper left")
axs[0].set_title('Tourism::ARIMA')
axs[0].set_xlabel('Time')

# deepar
axs[1].plot([i for i in range(292,367)], df_T[-75:], 'k-')
axs[1].plot([i for i in range(352,367)], deepar_params[:,0,0],'r-')
axs[1].fill_between(x = [i for i in range(352,367)], y1 = deepar_params[:,0,0]-2*np.sqrt(deepar_params[:,1,0]), y2 = deepar_params[:,0,0]+2*np.sqrt(deepar_params[:,1,0]), alpha=0.5)
axs[1].legend(["True", "Mean", "2_Sigma"], loc="upper left")
axs[1].set_title('Tourism::DeepAR')
axs[1].set_xlabel('Time')

plt.tight_layout()
plt.savefig('./Plot_new/Wiki/Base_Forecasts.png')
plt.show()
