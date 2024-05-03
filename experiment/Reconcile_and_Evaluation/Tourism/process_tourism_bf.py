import json
import util

with open('./Base_Forecasts/Tourism/Tourism.json','r') as file:
    data = json.load(file)

res = util.transform_fc(data,12)

with open('./Reconcile_and_Evaluation/Tourism/Tourism_out_process.json','w') as file2:
    file2.write(json.dumps(res))