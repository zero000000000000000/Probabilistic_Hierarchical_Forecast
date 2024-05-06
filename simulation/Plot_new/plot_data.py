import pandas as pd
import matplotlib.pyplot as plt

new_data = pd.read_csv('./Data/Simulated_Data_WithoutNoise.csv')
l1 = ['A','B']
l2 = ['AA','AB','BA','BB']
x = [i for i in range(1,(len(new_data)+1))]

fig,axs = plt.subplots(3,1,figsize = (12,9))
fig.subplots_adjust(hspace=0.5)

axs[0].plot(x,new_data['T'],label='T',color='black',linewidth=1)
axs[0].legend()
axs[0].set_title('Level_1')
axs[0].set_xlabel('Time')

colors = ['red','green','blue','yellow']
for i in range(2):
    axs[1].plot(x,new_data[l1[i]],label=l1[i],color=colors[i],linewidth=1)
axs[1].legend()
axs[1].set_title('Level_2')
axs[1].set_xlabel('Time')

for i in range(4):
    axs[2].plot(x,new_data[l2[i]],label=l2[i],color=colors[i],linewidth=1)
axs[2].legend()
axs[2].set_title('Level_3')
axs[2].set_xlabel('Time')


plt.savefig('./Plot_new/WithoutNoise_Data.png')
plt.show()