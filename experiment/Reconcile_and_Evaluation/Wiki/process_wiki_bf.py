import json
from util import transform_fc

with open('./Base_Forecasts/Wiki/Wiki.json','r') as file:
    data = json.load(file)

res = transform_fc(data,15)

with open('./Reconcile_and_Evaluation/Wiki/Wiki_out_process.json','w') as file2:
    file2.write(json.dumps(res))