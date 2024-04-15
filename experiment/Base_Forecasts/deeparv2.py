import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.evaluation import make_evaluation_predictions
import matplotlib.pyplot as plt
from gluonts.torch.distributions import NormalOutput

##### Preprocess ######
# def get_region(s):
#     if len(s) < 3:
#         return 'MMM'
#     else:
#         return s[:3]

# def get_zone(s):
#     if len(s) < 2:
#         return 'MM'
#     else:
#         return s[:2]

# def get_state(s):
#     if s == 'T':
#         return 'M'
#     else:
#         return s[0]

# # Preprocess the data
# df = pd.read_csv('d:\HierarchicalCode\experiment\Data\Tourism\Tourism_process.csv')

# level_1,level_2,level_3 = [],[],[]
# tags = [i for i in df.columns[2:]]
# for i in tags:
#     if (len(i) == 1) & (i != 'T'):
#         level_1.append(i)
#     elif len(i) == 2:
#         level_2.append(i)
#     elif len(i) == 3:
#         level_3.append(i)


# df['Year'] = df['Year'].astype(int)
# df['Year'] = df['Year'].astype(str)
# df['Month'] = df['Month'].astype(int)
# df['Month'] = df['Month'].astype(str)
# df['Date'] = df['Year']+"-"+df['Month']
# df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
# #df = df.set_index('Date')
# df.drop(['Year','Month'],axis=1,inplace=True)
# node_list = list(df.columns[:-1])
# df1 = pd.melt(df, id_vars=['Date'], value_vars=node_list, var_name='Node', value_name='Value')
# df1['Node'] = pd.Categorical(df1['Node'], categories=node_list, ordered=True)
# df2 = df1.sort_values(by=['Date', 'Node'], ascending=[True, True])
# df2.set_index('Date',inplace=True)

# # Add covariates
# df2['Month_Of_Year'] = pd.DatetimeIndex(df2.index).month
# df2['Region'] = df2['Node'].apply(get_region)
# lis1 = ['MMM']
# lis1.extend(level_3)
# df2['Region'] = pd.Categorical(df2['Region'], categories=lis1, ordered=True).codes
# df2['Zone'] = df2['Node'].apply(get_zone)
# lis2 = ['MM']
# lis2.extend(level_2)
# df2['Zone'] = pd.Categorical(df2['Zone'], categories=lis2, ordered=True).codes
# df2['State'] = df2['Node'].apply(get_state)
# lis3 = ['M']
# lis3.extend(level_1)
# df2['State'] = pd.Categorical(df2['State'], categories=lis3, ordered=True).codes
# df2.to_csv('./Base_Forecasts/Tourism_process_for_deepar.csv',index=True)

# Split train and test
df = pd.read_csv('./Base_Forecasts/Tourism_process_for_deepar.csv')
df.set_index('Date',inplace=True)
#df.index = pd.to_datetime(df.index).dt.strftime('%Y-%m-%d %H:%M:%S')
prediction_length = 12
freq = 'M'
split_date = pd.to_datetime('2016-01-01').strftime('%Y-%m-%d %H:%M:%S')
train = df[df.index < split_date]
test = df[df.index >= split_date]
train.reset_index(inplace=True)
test.reset_index(inplace=True)
# train1 = train.drop_duplicates(subset=['Node','Region','Zone','State'])
# train_static = pd.DataFrame({'State':train1['State'],
#                              'Zone':train1['Zone'],
#                              'Region':train1['Region'],
#                              'Node':train1['Node']})
# train_static.set_index('Node',inplace=True)
train_ds = PandasDataset.from_long_dataframe(train.iloc[:,[0,1,2]],
                                             target="Value",
                                             item_id="Node",
                                             timestamp='Date')

# Estimator
estimator = DeepAREstimator(freq=freq,
                            prediction_length=prediction_length,
                            context_length=4*prediction_length,
                            num_layers=3,
                            scaling=False,
                            # hidden_size=40,
                            # num_feat_dynamic_real=0,
                            # num_feat_static_cat=0,
                            # num_feat_static_real=0,
                            distr_output = NormalOutput(),
                            trainer_kwargs={'accelerator':'cpu','max_epochs':5})

# Train
predictor = estimator.train(train_ds,num_workers=4)

# Evaluation
# test1 = test.drop_duplicates(subset=['Node','Region','Zone','State'])
# test_static = pd.DataFrame({'State':test1['State'],
#                              'Zone':test1['Zone'],
#                              'Region':test1['Region'],
#                              'Node':test1['Node']})
# test_static.set_index('Node',inplace=True)

test_ds = PandasDataset.from_long_dataframe(test.iloc[:,[0,1,2]],
                                             target="Value",
                                             item_id="Node",
                                             timestamp='Date')
forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds, predictor=predictor,num_samples=100)
forecasts = list(forecast_it)
tests = list(ts_it)

n_plot = 3
indices = np.random.choice(np.arange(0, 111), size=n_plot, replace=False)
fig, axes = plt.subplots(n_plot, 1, figsize=(10, n_plot * 5))
for index, ax in zip(indices, axes):
    ax.plot(tests[index].to_timestamp())
    plt.sca(ax)
    forecasts[index].plot(intervals=(0.9,), color="g")
    plt.legend(["observed", "predicted median", "90% prediction interval"])