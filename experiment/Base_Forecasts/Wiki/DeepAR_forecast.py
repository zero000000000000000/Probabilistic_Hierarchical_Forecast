import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput
#from gluonts.torch.distributions import NormalOutput
from gluonts.evaluation import make_evaluation_predictions
import matplotlib.pyplot as plt
import json

# Split train and test
df = pd.read_csv('./Base_Forecasts/Wiki/Wiki_process_for_deepar.csv')
df.set_index('Date',inplace=True)
#df.index = pd.to_datetime(df.index).dt.strftime('%Y-%m-%d %H:%M:%S')
prediction_length = 15
freq = 'D'
split_date = pd.to_datetime('2016-12-17').strftime('%Y-%m-%d %H:%M:%S')
train = df[df.index < split_date]
test = df[df.index >= split_date]

train.reset_index(inplace=True)
test.reset_index(inplace=True)

train1 = train.drop_duplicates(subset=['Node','Code','Agent','Access','State'])
train_static = pd.DataFrame({'State':train1['State'],
                             'Access':train1['Access'],
                             'Agent':train1['Agent'],
                             'Code':train1['Code'],
                             'Node':train1['Node']})
train_static.set_index('Node',inplace=True)

# train_group = train.groupby('Node')
# standardized_params = {}
# train_standard = train.iloc[:,:]
# for cat,group in train_group:
#     train_group = train.groupby('Node')
#     means = group['Value'].mean()
#     stds = group['Value'].std()
#     standardized_params[cat] = {'mean':means,'std':stds}

#     train_standard.loc[group.index,'Value'] = (group['Value']-means)/stds


train_ds = PandasDataset.from_long_dataframe(train.iloc[:,[0,1,2,3,4,5]],
                                             target="Value",
                                             timestamp='Date',
                                             freq='D',
                                             item_id="Node",
                                             feat_dynamic_real=['Day_of_Week','Day_Of_Month','Month_Of_Year'],
                                             static_features=train_static)

# Estimator
estimator = DeepAREstimator(freq=freq,
                            prediction_length=prediction_length,
                            context_length=prediction_length,
                            num_layers=2,
                            hidden_size=40,
                            lr=1e-3,
                            weight_decay=1e-8,
                            dropout_rate=0.1,
                            num_feat_dynamic_real=3,
                            num_feat_static_real=4,
                            distr_output=NegativeBinomialOutput(),
                            scaling=False,
                            num_parallel_samples=1000,
                            batch_size=32,
                            trainer_kwargs={'accelerator':'gpu','max_epochs':100})

# Train
predictor = estimator.train(train_ds,num_workers=4)

# Evaluation
test1 = test.drop_duplicates(subset=['Node','Code','Agent','Access','State'])
test_static = pd.DataFrame({'State':test1['State'],
                             'Access':test1['Access'],
                             'Agent':test1['Agent'],
                             'Code':test1['Code'],
                             'Node':test1['Node']})
test_static.set_index('Node',inplace=True)

test_ds = PandasDataset.from_long_dataframe(test.iloc[:,[0,1,2,3,4,5]],
                                             target="Value",
                                             item_id="Node",
                                             feat_dynamic_real=['Day_of_Week','Day_Of_Month','Month_Of_Year'],
                                             static_features=test_static,
                                             timestamp='Date',
                                             freq='D')

forecast_it, ts_it = make_evaluation_predictions(dataset=test_ds, predictor=predictor,num_samples=1000)
forecasts = list(forecast_it)
tests = list(ts_it)

# # Anti standardized
# f_cp = forecasts
# for index in range(len(forecasts)):
#     index_mean = standardized_params[forecasts[index].item_id]['mean']
#     index_std = standardized_params[forecasts[index].item_id]['std']
#     print(index_mean)
#     print(index_std)
#     print(forecasts[index].item_id)
#     f_cp[index].samples = (forecasts[index].samples)*index_std+index_mean


# Plot
n_plot = 3
indices = np.random.choice(np.arange(0, 199), size=n_plot, replace=False)
fig, axes = plt.subplots(n_plot, 1, figsize=(10, n_plot * 5))
for index, ax in zip(indices, axes):
    ax.plot(tests[index].to_timestamp())
    plt.sca(ax)
    print(forecasts[index].item_id)
    forecasts[index].plot(intervals=(0.9,), color="g")
    plt.legend(["observed", "predicted median", "90% prediction interval"])

# sort
node_nonsort = []
new_df = pd.read_csv('d:\HierarchicalCode\experiment\Data\Wiki\Wiki_process.csv')
node_list = new_df.columns[1:]
for i in range(len(forecasts)):
    node_nonsort.append(forecasts[i].item_id)
index_list = [node_nonsort.index(i) for i in node_list]

# Save the distri params
distr_params = []
for j in range(15):
    params_mean = []
    params_var = []
    for i in range(len(forecasts)):
        params_mean.append(float(np.mean(forecasts[i].samples[:,j])))
        params_var.append(float(np.var(forecasts[i].samples[:,j],ddof=1)))
    distr_params.append([params_mean,params_var])

distr_params_new = distr_params
for j in range(15):
    distr_params_new[j][0] = [distr_params[j][0][i] for i in index_list]
    distr_params_new[j][1] = [distr_params[j][1][i] for i in index_list]
    
with open('./Base_Forecasts/Wiki/Wiki_deepar.json','w') as file:
    file.write(json.dumps(distr_params_new))