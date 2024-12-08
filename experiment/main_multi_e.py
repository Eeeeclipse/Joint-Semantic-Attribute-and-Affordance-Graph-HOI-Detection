from torch_geometric.data import HeteroData
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from experiment.model_multi_e import HELPv3e
from torch.utils.tensorboard import SummaryWriter 
summary_writer = SummaryWriter('tensorboard')
# torch.manual_seed(128)

dataset_path = 'hico_20160224_det'

with open(os.path.join(dataset_path, "action_list.json"), 'r') as file:
    action_list = json.load(file)

with open(os.path.join(dataset_path, "object_list.json"), 'r') as file:
    object_list = json.load(file)

def generate_train_graph_data(train_raw):
    HOI_graphs = []
    for graph in train_raw:
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
        # 在多节点模式下，因为无法确定label究竟来自哪条边，所以edge_label实际上指的是graph_label，但是这里就将错就错了
        act_label = F.one_hot(torch.tensor(graph['edge_label'][0]), num_classes=118)
        obj_label = F.one_hot(torch.tensor(graph['edge_label'][1]), num_classes=118)
        data.y = torch.stack((act_label, obj_label))[None,:,:]
        HOI_graphs.append(data)
    return HOI_graphs


def train(train_data, learning_rate, batch_size, epochs, device):
    best_model = None
    min_loss = None
    model = HELPv3e(num_edge_classes=118)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  # Adjust batch size as needed
    total_batches = len(loader)
    model.train()
    for epoch in range(epochs):
        print
        total_loss = 0
        bidx = 0
        for batch in tqdm(loader):
            batch = batch.to(device)
            edge_labels = batch.y.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch['human'].batch, batch['object'].batch)
            loss = criterion(out, F.softmax(edge_labels.to(torch.float32), dim=1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            summary_writer.add_scalar('loss_batch', loss.item(), epoch * total_batches + bidx)
            bidx+=1
        avg_loss = total_loss / len(loader)
        summary_writer.add_scalar('loss_epoch', avg_loss, epoch)
        if min_loss == None or avg_loss < min_loss:
            best_model = model
        print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    return best_model


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

    # Fast test
    learning_rate = 5e-4
    epochs = 20

    # learning_rate = 1e-5
    # epochs = 200 

    batch_size = 256
    device = 'cuda'
    with open("graph_train_multi.json", 'r') as file:
        train_raw = json.load(file)
    train_data = generate_train_graph_data(train_raw)
    best_model = train(train_data, learning_rate, batch_size, epochs, device)
    torch.save(best_model.state_dict(), "models/ResGat_m.pth")

    with open("graph_test_multi.json", 'r') as file:
        test_raw = json.load(file)

    test_data, id_dict = generate_test_graph_data(test_raw)

    test_accu = evaluation(best_model, test_data, id_dict, batch_size, device)

    print('-' * 80)
    print(best_model)
    print('Accuracy on test dataset:')
    print(test_accu)
    print('-' * 80)

