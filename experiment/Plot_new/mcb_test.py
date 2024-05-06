import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./Analyze_Result_new/CRPS/Tourism_ETS_mean_crps.csv').iloc[:,1:]
df.columns = ['Base','BottomUp','JPP','OLS','OLSv','WLS','WLSv','MinTShr','MinTShrv','EnergyScore_Opt']

rank_res = df.rank(axis=1)
k=10
N=111
r = 4.743
d = k*(k+1)/(12*N)
x = rank_res.mean().sort_values()
y = list(range(0,k,1))
xerr = d**0.5*r

# plot:
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_facecolor('white')
ax.errorbar(x, y, xerr = xerr, fmt='o', linewidth=2, capsize=6)
ax.set_yticks(list(range(0,k,1)))
ax.set_yticklabels(list(x.index))
ax.set_title('Tourism::ETS')
plt.savefig('./Plot_new/Tourism/MCB_Test_ETS.png')
plt.show()