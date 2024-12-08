import scipy.io as scio
import random
import os
import json 

random.seed(10)
dataset_path = './hico_20160224_det'

with open(os.path.join(dataset_path, "actions.json"), 'r') as file:
    action_dict = json.load(file)

act_list = []
obj_list = []
for key in action_dict.keys():
    act, obj = action_dict[key]
    if act not in act_list:
        act_list.append(act)
    if obj not in obj_list:
        obj_list.append(obj)

act_list = sorted(act_list)
obj_list = sorted(obj_list)

print('Length of action list', len(act_list))
print('Length of object list', len(obj_list))
with open(os.path.join(dataset_path, "action_list.json"), "w") as action_file: 
    json.dump(act_list, action_file)
with open(os.path.join(dataset_path, "object_list.json"), "w") as obj_file: 
    json.dump(obj_list, obj_file)