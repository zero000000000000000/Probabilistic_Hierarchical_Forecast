import json
import util

with open('./Base_Forecasts/Labour/labour.json','r') as file:
    data = json.load(file)

res = util.transform_fc(data,6)

with open('./Reconcile_and_Evaluation/Labour/labour_out_process.json','w') as file2:
    file2.write(json.dumps(res))