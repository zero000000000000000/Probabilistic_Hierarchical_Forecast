import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# Read raw data
df = pd.read_csv('./Data/Tourism/Tourism_process.csv')
df_T = df['T'].tolist()

# Read the params
with open('./Reconcile_and_Evaluation/Tourism/Tourism_out_process.json','r') as file1:
    params_ets = json.load(file1)

params_ets_1 = params_ets[0]
ets_params = []
for i in params_ets_1:
    ets_params.append(i[:2])
ets_params = np.array(ets_params)

with open('./Base_Forecasts/Tourism/Tourism_deepar_e300_minmax.json','r') as file2:
    params_deepar = json.load(file2)

deepar_params = np.array(params_deepar)

fig,axs = plt.subplots(2,1,figsize = (12,9))

# ets
axs[0].plot([i for i in range(169,229)],df_T[-60:], 'k-')
axs[0].plot([i for i in range(217,229)], ets_params[:,0,0],'r-')
axs[0].fill_between(x = [i for i in range(217,229)], y1 = ets_params[:,0,0]-2*np.sqrt(ets_params[:,1,0]), y2 = ets_params[:,0,0]+2*np.sqrt(ets_params[:,1,0]), alpha=0.5)
axs[0].legend(["True", "Mean", "2_Sigma"], loc="upper left")
axs[0].set_title('Tourism::ETS')
axs[0].set_xlabel('Time')

# deepar
axs[1].plot([i for i in range(169,229)], df_T[-60:], 'k-')
axs[1].plot([i for i in range(217,229)], deepar_params[:,0,0],'r-')
axs[1].fill_between(x = [i for i in range(217,229)], y1 = deepar_params[:,0,0]-2*np.sqrt(deepar_params[:,1,0]), y2 = deepar_params[:,0,0]+2*np.sqrt(deepar_params[:,1,0]), alpha=0.5)
axs[1].legend(["True", "Mean", "2_Sigma"], loc="upper left")
axs[1].set_title('Tourism::DeepAR')
axs[1].set_xlabel('Time')

plt.tight_layout()
plt.savefig('./Plot_new/Tourism/Base_Forecasts_minmax.png')
plt.show()
