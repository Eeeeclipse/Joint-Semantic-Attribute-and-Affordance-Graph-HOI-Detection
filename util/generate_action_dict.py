import scipy.io as scio
import random
import os
import json 

random.seed(10)
dataset_path = './hico_20160224_det'
annoFile = 'anno.mat'
data = scio.loadmat(os.path.join(dataset_path, annoFile))
action_dict = dict()
idx = 1
for aa in data['list_action']:
    aa = aa[0]
    obj = aa[0][0]
    act = aa[1][0]
    action_dict[str(idx)] = [act, obj]
    idx += 1
with open(os.path.join(dataset_path, "actions.json"), "w") as action_file: 
    json.dump(action_dict, action_file)