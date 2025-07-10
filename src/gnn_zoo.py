import logging
import pickle

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn import GATConv, GINConv, SGConv, SAGEConv
from torch_geometric.loader import NeighborSampler, NeighborLoader

import numpy as np
from typing import Union
from torch_geometric.typing import Adj, OptTensor, OptPairTensor


class GNNBase:
    def __init__(self):
        self.logger = logging.getLogger('gnn')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.embedding_dim = 0
        self.data = None
        self.subgraph_loader = None

    def save_model(self, save_path):
        self.logger.info('saving model')
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        self.logger.info('loading model')
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(save_path, map_location=device))

    def save_paras(self, save_path):
        self.logger.info('saving paras')
        self.paras = {
            'embedding_dim': self.embedding_dim
        }
        pickle.dump(self.paras, open(save_path, 'wb'))

    def load_paras(self, save_path):
        self.logger.info('loading paras')
        return pickle.load(open(save_path, 'rb'))

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def posterior(self):
        self.model.eval()
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)

        posteriors = self.model(self.data)
        for _, mask in self.data('test_mask'):
            posteriors = posteriors[mask]

        return posteriors.detach()



class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)
    
    def forward_once_unlearn(self, data):
        for i in range(self.num_layers - 1):
            x = self.convs[i](data.x_unlearn, data.edge_index_unlearn)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        # Output layer
        x = self.convs[-1](data.x_unlearn, data.edge_index_unlearn)
        return F.log_softmax(x, dim=-1)
    
        """
        x = F.dropout(data.x_unlearn, p=self.dropout, training=self.training)
        x = F.relu(self.convs[0](x, data.edge_index_unlearn))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, data.edge_index_unlearn)

        return F.log_softmax(x, dim=1)
        """

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)

                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class ClassMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(ClassMLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels,bias=False))
            self.dropout = dropout
            return
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels,bias=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels,bias=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels,bias=False))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)
        # return x
        
class GATNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, dropout=0.6):
        super(GATNet, self).__init__()
        self.dropout = dropout

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_feats, 8, heads=8, dropout=self.dropout, add_self_loops=True))
        # On the Pubmed dataset, use heads=8 in conv2.
        self.convs.append(GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=self.dropout, add_self_loops=True))

    def forward_v0(self, x, adjs):
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.convs[0](x, edge_index)) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def forward_once(self, data):
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = F.relu(self.convs[0](x, data.edge_index)) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, data.edge_index)

        return F.log_softmax(x, dim=1)

    def forward_once_unlearn(self, data):
        x = F.dropout(data.x_unlearn, p=self.dropout, training=self.training)
        x = F.relu(self.convs[0](x, data.edge_index_unlearn))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[1](x, data.edge_index_unlearn)

        return F.log_softmax(x, dim=1)

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)

                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()





class GCNConvBatch(GCNConv):
    def __init__(self, in_channels: int, out_channels: int = 16,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, bias: bool = True, 
                 **kwargs):
        super(GCNConvBatch, self).__init__(in_channels, out_channels,
                                           improved=improved, cached=cached, add_self_loops=add_self_loops,
                                           bias=bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = self.lin(out)

        return out

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, linear_only = False):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.linear_only = linear_only
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_channels) 
            for _ in range(num_layers-1)
        ])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            
            if(len(self.batch_norms)):
                x = self.batch_norms[i](x)
            
            if(self.linear_only == False):
                x = F.relu(x)
                
            if(self.dropout < 1):
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if(len(self.convs) > 1):
            x = self.convs[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)
        #return x # otherwise, nll_loss will be negative.
    
    def inference(self, x_all, subgraph_loader, edge_weight, device):
        for i, conv in enumerate(self.convs[:-1]):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = F.relu(x)

                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        x_all = self.convs[-1](x_all, subgraph_loader)

        return x_all
    
    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()

class GCNNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, hidden_channels = 256, dropout = 0, batch_norm = False, linear_only = False):
        super(GCNNet, self).__init__()

        self.num_layers = 2
        self.linear_only = linear_only
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConvBatch(num_feats, out_channels=hidden_channels, cached=False, add_self_loops=True, bias=False))
        self.convs.append(GCNConvBatch(in_channels=hidden_channels, out_channels=num_classes, cached=False, add_self_loops=True, bias=False))

        self.dropout = dropout
        if(batch_norm == True):
            self.batch_norm = torch.nn.ModuleList()
            self.batch_norm.append(torch.nn.BatchNorm1d(hidden_channels))
            self.batch_norm.append(torch.nn.BatchNorm1d(num_classes))
            
    def forward(self, x, adjs, edge_weight):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

    def forward_once(self, data, edge_weight = None):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index, edge_weight)
        
        if(self.linear_only == False):
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, data, edge_weight = None):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x, edge_index, edge_weight)
        
        if(self.linear_only == False):
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, edge_weight, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()



class GINNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(GINNet, self).__init__()

        dim = 32
        self.num_layers = 2

        nn1 = Sequential(Linear(num_feats, dim), ReLU(), Linear(dim, dim))
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))

        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn1))
        self.convs.append(GINConv(nn2))

        self.bn = torch.nn.ModuleList()
        self.bn.append(torch.nn.BatchNorm1d(dim))
        self.bn.append(torch.nn.BatchNorm1d(dim))

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward_v0(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

            x = self.bn[i](x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def forward(self, x, edge_index):
        x = F.relu(self.convs[0](x, edge_index))
        x = self.bn[0](F.dropout(x, p=0.5, training=self.training))
        x = self.convs[1](x, edge_index)
        x = self.bn[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    def forward_once(self, data):
        x = F.relu(self.convs[0](data.x, data.edge_index))
        x = self.bn[0](F.dropout(x, p=0.5, training=self.training))
        x = self.convs[1](x, data.edge_index)
        x = self.bn[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def forward_once_unlearn(self, data):
        x = F.relu(self.convs[0](data.x_unlearn, data.edge_index_unlearn))
        x = self.bn[0](F.dropout(x, p=0.5, training=self.training))
        x = self.convs[1](x, data.edge_index_unlearn)
        x = self.bn[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)

                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)

                if i != self.num_layers - 1:
                    x = F.relu(x)

                x = self.bn[i](x)

                xs.append(x)

            x_all = torch.cat(xs, dim=0)

        x_all = F.relu(self.fc1(x_all))
        x_all = self.fc2(x_all)

        return x_all.cpu()

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()





class SGConvBatch(SGConv):
    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True, 
                 bias: bool = True, **kwargs):
        super(SGConvBatch, self).__init__(in_channels, out_channels,
                                          cached=cached, add_self_loops=add_self_loops,
                                          bias=bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = self.lin(out)

        return out



class SGCNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes):
        super(SGCNet, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SGConvBatch(num_feats, 16, cached=False, add_self_loops=True, bias=False))
        self.convs.append(SGConvBatch(16, num_classes, cached=False, add_self_loops=True, bias=False))

    def forward(self, x, adjs, edge_weight):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

            if i != self.num_layers - 1:
                x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

    def forward_once_v0(self, data, edge_weight = None):
        x, edge_index = data.x, data.edge_index
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once(self, x, edge_index, edge_weight = None):
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)
    
    def forward_once_unlearn_v0(self, data, edge_weight):
        x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)

    def forward_once_unlearn(self, x, edge_index, edge_weight = None):
        #x, edge_index = data.x_unlearn, data.edge_index_unlearn
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)

        return F.log_softmax(x, dim=-1)
    
    def inference(self, x_all, subgraph_loader, edge_weight, device):
        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index, edge_weight=edge_weight[e_id])

                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()

