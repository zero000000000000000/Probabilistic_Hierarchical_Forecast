import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimHei'],
}
plt.rcParams.update(config)


# Read raw data
df = pd.read_csv('./Data/Labour/labour_process_place.csv')
df_T = df['T'].tolist()

# Read the params
with open('./Reconcile_and_Evaluation/Labour/labour_out_process.json','r') as file1:
    params_ets = json.load(file1)

ets_params = []
for i in params_ets:
    ets_params.append(i[:2])
ets_params = np.array(ets_params)

# with open('./Base_Forecasts/Tourism/Tourism_deepar_e300_optuna.json','r') as file2:
#     params_deepar = json.load(file2)

# deepar_params = np.array(params_deepar)

plt.figure(figsize = (12,9))

# ets
plt.plot([i for i in range(134,164)],df_T[-30:], 'k-')
plt.plot([i for i in range(158,164)], ets_params[:,0,0],'r-')
plt.fill_between(x = [i for i in range(158,164)], y1 = ets_params[:,0,0]-2*np.sqrt(ets_params[:,1,0]), y2 = ets_params[:,0,0]+2*np.sqrt(ets_params[:,1,0]), alpha=0.5)
plt.legend(["真实值", "预测均值", "$\mathrm{2Sigma}$预测区间"], loc="upper left")
plt.title('$\mathrm{Labour::ARIMA}$')
plt.xlabel('时间')

# # deepar
# axs[1].plot([i for i in range(169,229)], df_T[-60:], 'k-')
# axs[1].plot([i for i in range(217,229)], deepar_params[:,0,0],'r-')
# axs[1].fill_between(x = [i for i in range(217,229)], y1 = deepar_params[:,0,0]-2*np.sqrt(deepar_params[:,1,0]), y2 = deepar_params[:,0,0]+2*np.sqrt(deepar_params[:,1,0]), alpha=0.5)
# axs[1].legend(["真实值", "预测均值", "2Sigma预测区间"], loc="upper left")
# axs[1].set_title('Tourism::DeepAR')
# axs[1].set_xlabel('Time')

plt.tight_layout()

plt.savefig('./Plot_new/Labour/Base_Forecasts_labour.png')
plt.show()