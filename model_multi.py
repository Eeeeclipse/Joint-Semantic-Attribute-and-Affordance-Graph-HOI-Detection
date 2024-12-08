import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool
import torch.nn.functional as F

# HELP stands for "Hetero Edge Label Predictor"
class HELPv3(torch.nn.Module):
    def __init__(self, num_edge_classes):
        super().__init__()
        cm0_dim = 256
        cm0_num = 8
        self.conv_module_0 = []
        for i in range(cm0_num):
            conv = HeteroConv({
                ('human', 'interact', 'object'): GATConv(in_channels=(-1, -1), out_channels=cm0_dim, edge_dim=2, add_self_loops=False),
                ('object', 'interacted_by', 'human'): GATConv(in_channels=(-1, -1), out_channels=cm0_dim, edge_dim=2, add_self_loops=False),
            }, aggr='sum')
            self.conv_module_0.append(conv)
        self.conv_module_0 = torch.nn.ModuleList(self.conv_module_0)
        self.edge_mlp1 = Linear(-1, 32)
        self.pred_mlp1 = Linear(-1, num_edge_classes)
        self.pred_mlp2 = Linear(-1, num_edge_classes)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, hbatch, obatch):
        x_dict_0 = None
        for conv in self.conv_module_0:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            if x_dict_0 == None:
                x_dict = {k:F.relu(x) for k, x in x_dict.items()}
            else:
                x_dict = {k:F.relu(x) + x_dict_0[k] for k, x in x_dict.items()}
            x_dict_0 = x_dict
        for k, x in x_dict.items():
            if k == 'human':
                x_dict[k] = global_mean_pool(x, hbatch)
            elif k == 'object':
                x_dict[k] = global_mean_pool(x, obatch)
            else:
                assert False
        edge_attr = edge_attr_dict['human', 'interact', 'object']
        edge_attr = F.relu(self.edge_mlp1(edge_attr))
        concat_idx = torch.unique(hbatch)
        human_emb = x_dict['human'][concat_idx]
        object_emb = x_dict['object'][concat_idx]
        h, o = edge_index_dict['human', 'interact', 'object']
        edge_batch = hbatch[h]
        edge_attr = global_mean_pool(edge_attr, edge_batch)
        graph_emb = torch.cat([human_emb, object_emb, edge_attr], dim=-1)
        graph_emb = F.dropout(graph_emb, p=0.2, training=self.training) 
        edge_logits1 = F.softmax(self.pred_mlp1(graph_emb), dim=1)
        edge_logits2 = F.softmax(self.pred_mlp2(graph_emb), dim=1)
        out = torch.stack((edge_logits1, edge_logits2), axis = 1)
        return out