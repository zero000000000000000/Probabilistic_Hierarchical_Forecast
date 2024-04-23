import pandas as pd

##### Preprocess ######

def get_code(s):
    if s.count('_') == 3:
        return s
    else:
        return "MMMM"


def get_agent(s):
    if s.count('_') == 3:
        ss = s.split('_')
        ss = '_'.join(ss[:3])
        return ss
    elif s.count('_') == 2:
        return s
    else:
        return 'MMM'

def get_access(s):
    if s.count('_') == 3:
        ss = s.split('_')
        ss = '_'.join(ss[:2])
        return ss
    elif s.count('_') == 2:
        ss = s.split('_')
        ss = '_'.join(ss[:2])
        return ss
    elif s.count('_') == 1:
        return s
    else:
        return 'MM'


def get_state(s):
    if s == 'Total':
        return 'M'
    elif s.count('_') == 0:
        return s
    else:
        ss = s.split('_')
        return ss[0]

# Preprocess the data
df = pd.read_csv('d:\HierarchicalCode\experiment\Data\Wiki\Wiki_process.csv')

level_1,level_2,level_3,level_4= [],[],[],[]
tags = [i for i in df.columns[1:]]
for i in tags:
    if (i.count('_') == 0) & (i != 'Total'):
        level_1.append(i)
    elif i.count('_') == 1:
        level_2.append(i)
    elif i.count('_') == 2:
        level_3.append(i)
    elif i.count('_') == 3:
        level_4.append(i)


# df['Year'] = df['Year'].astype(int)
# df['Year'] = df['Year'].astype(str)
# df['Month'] = df['Month'].astype(int)
# df['Month'] = df['Month'].astype(str)
# df['Date'] = df['Year']+"-"+df['Month']
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
#df = df.set_index('Date')
# df.drop(['Year','Month'],axis=1,inplace=True)
node_list = list(df.columns[1:])

df1 = pd.melt(df, id_vars=['Date'], value_vars=node_list, var_name='Node', value_name='Value')
df1['Node'] = pd.Categorical(df1['Node'], categories=node_list, ordered=True)
df2 = df1.sort_values(by=['Date', 'Node'], ascending=[True, True])
df2.set_index('Date',inplace=True)

# Add covariates
df2['Day_of_Week'] = pd.DatetimeIndex(df2.index).dayofweek
df2['Day_Of_Month'] = pd.DatetimeIndex(df2.index).day
df2['Month_Of_Year'] = pd.DatetimeIndex(df2.index).month

df2['Code'] = df2['Node'].apply(get_code)
lis0 = ['MMMM']
lis0.extend(level_4)
df2['Code'] = pd.Categorical(df2['Code'], categories=lis0, ordered=True).codes

df2['Agent'] = df2['Node'].apply(get_agent)
lis1 = ['MMM']
lis1.extend(level_3)
df2['Agent'] = pd.Categorical(df2['Agent'], categories=lis1, ordered=True).codes

df2['Access'] = df2['Node'].apply(get_access)
lis2 = ['MM']
lis2.extend(level_2)
df2['Access'] = pd.Categorical(df2['Access'], categories=lis2, ordered=True).codes

df2['State'] = df2['Node'].apply(get_state)
lis3 = ['M']
lis3.extend(level_1)
df2['State'] = pd.Categorical(df2['State'], categories=lis3, ordered=True).codes

df2.to_csv('./Base_Forecasts/Wiki/Wiki_process_for_deepar.csv',index=True)