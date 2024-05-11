import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者使用 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['font.size'] = 15

new_data = pd.read_csv('./Data/Simulated_Data_WithoutNoise.csv')
l1 = ['A','B']
l2 = ['AA','AB','BA','BB']
x = [i for i in range(1,(len(new_data)+1))]

fig,axs = plt.subplots(3,1,figsize = (12,9))
fig.subplots_adjust(hspace=0.5)

axs[0].plot(x,new_data['T'],label='T',color='black',linewidth=1)
axs[0].legend()
axs[0].set_title('层次1')
axs[0].set_xlabel('时间')

colors = ['red','green','blue','yellow']
for i in range(2):
    axs[1].plot(x,new_data[l1[i]],label=l1[i],color=colors[i],linewidth=1)
axs[1].legend()
axs[1].set_title('层次2')
axs[1].set_xlabel('时间')

for i in range(4):
    axs[2].plot(x,new_data[l2[i]],label=l2[i],color=colors[i],linewidth=1)
axs[2].legend()
axs[2].set_title('层次3')
axs[2].set_xlabel('时间')

plt.tight_layout()
plt.savefig('./Plot_new/WithoutNoise_Data.png')
plt.show()