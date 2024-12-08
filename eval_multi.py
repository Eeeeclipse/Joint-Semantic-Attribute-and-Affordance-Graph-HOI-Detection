from torch_geometric.data import HeteroData
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model_multi import HELPv3

def generate_test_graph_data(test_raw):
    id_dict = dict()
    HOI_graphs = []
    for graph in test_raw:
        data = HeteroData()
        human = []
        for human_id in graph['human_feature'].keys():
            human.append(graph['human_feature'][human_id][0])
        data['human'].x = torch.tensor(human)
        
        object = []
        for obj_id in graph['obj_feature'].keys():
            attr = graph['obj_feature'][obj_id][0][0]
            aff = graph['obj_feature'][obj_id][1][0]
            object.append(attr + aff)
        data['object'].x = torch.tensor(object)

        h = []
        o = []
        edge_feature_ho = []
        for edge_id in graph['edge_feature'].keys():
            human_id, obj_id = edge_id.split(';')
            h.append(int(human_id))
            o.append(int(obj_id))
            edge_feature_ho.append(graph['edge_feature'][edge_id])
        data['human', 'interact', 'object'].edge_index = torch.tensor([h, o])
        data['human', 'interact', 'object'].edge_attr = torch.tensor(edge_feature_ho)
        
        data['object', 'interacted_by', 'human'].edge_index = torch.tensor([o,h])
        data['object', 'interacted_by', 'human'].edge_attr = -torch.tensor(edge_feature_ho)
        data['id'] = graph['id']
        # 在多节点模式下，因为无法确定label究竟来自哪条边，所以edge_label实际上指的是graph_label，但是这里就将错就错了
        id_dict[graph['id']] = graph['labels']
        HOI_graphs.append(data)
    return HOI_graphs, id_dict

def evaluation(model, test_data, id_dict, batch_size, device):
    count = 0
    total = 0
    model = model.to(device)
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    for tbatch in tqdm(test_loader):
        tbatch = tbatch.to(device)

        with torch.no_grad():
            out = model(tbatch.x_dict, tbatch.edge_index_dict, tbatch.edge_attr_dict, tbatch['human'].batch, tbatch['object'].batch)
            predictions = out.argmax(dim=-1).tolist()
        for i in range(len(predictions)):
            total += 1
            tid = tbatch['id'][i]
            pred = predictions[i]
            gt = id_dict[tid]
            if pred in gt:
                count += 1
    return count / total

if __name__ == '__main__':
    device = 'cuda'
    weight = "models/ResGat_m.pth"
    batch_size = 1024

    best_model = HELPv3(num_edge_classes=118)
    best_model = best_model.to(device)
    GNN_weight = torch.load(weight)
    best_model.load_state_dict(GNN_weight)

    with open("graph_test_multi.json", 'r') as file:
        test_raw = json.load(file)

    test_data, id_dict = generate_test_graph_data(test_raw)

    test_accu = evaluation(best_model, test_data, id_dict, batch_size, device)

    print('-' * 80)
    print(best_model)
    print('Accuracy on test dataset:')
    print(test_accu)
    print('-' * 80)