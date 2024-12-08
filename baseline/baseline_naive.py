from torch_geometric.data import HeteroData
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, TransformerConv, SAGEConv, GATConv, Linear, SGConv
from torch_geometric.data import DataLoader
from matplotlib import pyplot as plt

with open("./graph_test_multi.json", 'r') as file:
    test_raw_multi = json.load(file)

count = 0
for to_test in test_raw_multi:
    ground_truth = to_test['labels']
    preds = []
    scores = []
    for human_id in to_test['human_feature'].keys():
        pose = torch.tensor(to_test['human_feature'][human_id][0])
        for obj_id in to_test['obj_feature'].keys():
            aff = torch.tensor(to_test['obj_feature'][obj_id][0])
            attr = torch.tensor(to_test['obj_feature'][obj_id][1])
            obj = attr[0]
            act = (aff + pose)[0]
            act_pred = act.argmax(dim=-1).tolist()
            obj_pred = obj.argmax(dim=-1).tolist()
            pred = [act_pred, obj_pred]
            score = act[act_pred] + obj[obj_pred]
            preds.append(pred)
            scores.append(score)


    pred_idx = np.argmax(scores)

    final_pred = preds[pred_idx]

    if final_pred in ground_truth:
        count += 1

print('-' * 80)
print('Naive baseline:')
print('Accuracy:')
print(count / len(test_raw_multi))
print('-' * 80)
