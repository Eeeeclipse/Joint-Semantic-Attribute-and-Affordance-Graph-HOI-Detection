import json
import os

dataset_path = './hico_20160224_det'

with open(os.path.join(dataset_path, "actions.json"), 'r')as file:
    action_pair = json.load(file)
action_pair_list = []
for k, x in action_pair.items():
    action_pair_str = x[0] + ' ' + x[1]
    action_pair_list.append(action_pair_str)

with open(os.path.join(dataset_path, "action_pair.json"), "w") as test_file: 
    json.dump(action_pair_list, test_file)