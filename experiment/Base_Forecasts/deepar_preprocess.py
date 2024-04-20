import pandas as pd

##### Preprocess ######
def get_region(s):
    if len(s) < 3:
        return 'MMM'
    else:
        return s[:3]

def get_zone(s):
    if len(s) < 2:
        return 'MM'
    else:
        return s[:2]

def get_state(s):
    if s == 'T':
        return 'M'
    else:
        return s[0]

# Preprocess the data
df = pd.read_csv('d:\HierarchicalCode\experiment\Data\Tourism\Tourism_process.csv')

level_1,level_2,level_3 = [],[],[]
tags = [i for i in df.columns[2:]]
for i in tags:
    if (len(i) == 1) & (i != 'T'):
        level_1.append(i)
    elif len(i) == 2:
        level_2.append(i)
    elif len(i) == 3:
        level_3.append(i)


df['Year'] = df['Year'].astype(int)
df['Year'] = df['Year'].astype(str)
df['Month'] = df['Month'].astype(int)
df['Month'] = df['Month'].astype(str)
df['Date'] = df['Year']+"-"+df['Month']
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
#df = df.set_index('Date')
df.drop(['Year','Month'],axis=1,inplace=True)
node_list = list(df.columns[:-1])
df1 = pd.melt(df, id_vars=['Date'], value_vars=node_list, var_name='Node', value_name='Value')
df1['Node'] = pd.Categorical(df1['Node'], categories=node_list, ordered=True)
df2 = df1.sort_values(by=['Date', 'Node'], ascending=[True, True])
df2.set_index('Date',inplace=True)

# Add covariates
df2['Month_Of_Year'] = pd.DatetimeIndex(df2.index).month
df2['Region'] = df2['Node'].apply(get_region)
lis1 = ['MMM']
lis1.extend(level_3)
df2['Region'] = pd.Categorical(df2['Region'], categories=lis1, ordered=True).codes
df2['Zone'] = df2['Node'].apply(get_zone)
lis2 = ['MM']
lis2.extend(level_2)
df2['Zone'] = pd.Categorical(df2['Zone'], categories=lis2, ordered=True).codes
df2['State'] = df2['Node'].apply(get_state)
lis3 = ['M']
lis3.extend(level_1)
df2['State'] = pd.Categorical(df2['State'], categories=lis3, ordered=True).codes
df2.to_csv('./Base_Forecasts/Tourism_process_for_deepar.csv',index=True)