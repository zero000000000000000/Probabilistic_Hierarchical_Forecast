import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.model.deepar import DeepAREstimator
#from gluonts.torch.distributions import NegativeBinomialOutput
from gluonts.torch.distributions import NormalOutput
import json

def Create_Data_Window(df,N,W,stride):
    '''
    Create W windows
    '''
    lis = []
    for i in range(W):
        lis.append(df.iloc[:(N+i*stride)*111,:].reset_index().iloc[:,1:])
    return lis

# Split train and test
df = pd.read_csv('./Base_Forecasts/Tourism_process_for_deepar.csv')
# df.set_index('Date',inplace=True)
prediction_length = 12
freq = 'MS'
stride = 12
N = 168
W = 4

train_lis = Create_Data_Window(df,N,W,stride)


train1 = train_lis[0].drop_duplicates(subset=['Node','Region','Zone','State'])
train_static = pd.DataFrame({'State':train1['State'],
                             'Zone':train1['Zone'],
                             'Region':train1['Region'],
                             'Node':train1['Node']})
train_static.set_index('Node',inplace=True)

sp = []
train_standard_lis = []
for train_ds in train_lis:
    train_group = train_ds.groupby('Node')
    standardized_params = {}
    train_standard = train_ds.copy()
    for cat,group in train_group:
        means = group['Value'].mean()
        stds = group['Value'].std()
        standardized_params[cat] = {'mean':means,'std':stds}
        train_standard.loc[group.index,'Value'] = (group['Value']-means)/stds
        train_standard_ds = PandasDataset.from_long_dataframe(train_standard.iloc[:,[0,1,2,3]],
                                                    target="Value",
                                                    timestamp='Date',
                                                    freq='M',
                                                    item_id="Node",
                                                    feat_dynamic_real=["Month_Of_Year"],
                                                    static_features=train_static)
        train_standard_lis.append(train_standard_ds)
    sp.append(standardized_params)

for i in range(len(train_standard_lis)):
    # Estimator
    estimator = DeepAREstimator(freq=freq,
                                prediction_length=prediction_length,
                                context_length=10*prediction_length,
                                num_layers=3,
                                hidden_size=41,
                                lr=1e-2,
                                weight_decay=1e-8,
                                dropout_rate=0.1,
                                num_feat_dynamic_real=1,
                                num_feat_static_real=3,
                                distr_output=NormalOutput(),
                                patience=10,
                                scaling=False,
                                num_parallel_samples=1000,
                                batch_size=16,
                                trainer_kwargs={'accelerator':'gpu','max_epochs':300})

    # Train
    predictor = estimator.train(train_standard_lis[i],num_workers=4)

    # Predict
    forecast_it = predictor.predict(train_standard_lis[i], num_samples = 1000)
    forecasts = list(forecast_it)


    # Anti standardized
    f_cp = forecasts
    for index in range(len(forecasts)):
        index_mean = standardized_params[i][forecasts[index].item_id]['mean']
        index_std = standardized_params[i][forecasts[index].item_id]['std']
        f_cp[index].samples = (forecasts[index].samples)*index_std+index_mean

    if i==0:
        # sort
        node_nonsort = []
        new_df = pd.read_csv('d:\HierarchicalCode\experiment\Data\Tourism\Tourism_process.csv')
        node_list = new_df.columns[2:]
        for i in range(len(forecasts)):
            node_nonsort.append(forecasts[i].item_id)
        index_list = [node_nonsort.index(i) for i in node_list]

    # Save the distri params
    distr_params = []
    for j in range(12):
        params_mean = []
        params_var = []
        for i in range(len(forecasts)):
            params_mean.append(float(np.mean(forecasts[i].samples[:,j])))
            params_var.append(float(np.var(forecasts[i].samples[:,j],ddof=1)))
        distr_params.append([params_mean,params_var])

    distr_params_new = distr_params
    for j in range(12):
        distr_params_new[j][0] = [distr_params[j][0][i] for i in index_list]
        distr_params_new[j][1] = [distr_params[j][1][i] for i in index_list]
        
    with open('./Base_Forecasts/Tourism_deepar_e300_optuna_in_{}.json'.format(i+1),'w') as file:
        file.write(json.dumps(distr_params_new))