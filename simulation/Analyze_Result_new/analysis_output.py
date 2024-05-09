import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generate', type=str, default='WithNoise', help='If there is added noise')
parser.add_argument('--rootbasef', type=str, default='ARIMA', help='The base forecast of root point')
parser.add_argument('--basefdep', type=str, default='Independent', help='The base forecast independence')

if __name__=='__main__':

    # Get params
    args = parser.parse_args()
    generate = args.generate
    rootbasef = args.rootbasef
    basefdep = args.basefdep

    df = pd.read_csv(f'./Analyze_Result_new/CRPS/{generate}_{rootbasef}_{basefdep}_mean_crps_v2.csv')
    l1 = df.iloc[0,1:].round(4)
    l2 = df.iloc[1:3,1:].mean(axis=0).round(4)
    l3 = df.iloc[3:7,1:].mean(axis=0).round(4)
    total = df.iloc[:,1:].mean(axis=0).round(4)
    # print(l1)
    # print(l2)
    # print(l3)
    df1 = pd.DataFrame({'l1':l1.tolist(),'l2':l2.tolist(),'l3':l3.tolist(),'T':total.tolist()})
    df1.to_csv(f'./Analyze_Result_new/output/{generate}_{rootbasef}_{basefdep}_mean_crps_v2.csv')

