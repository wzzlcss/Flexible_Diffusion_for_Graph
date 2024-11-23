import torch
import torch.nn as nn
import math
import torch.nn.functional as F

#########################################################
#########################################################
#########################################################
class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, n_layers, dropout, criterion):
        # provable benefit implementation: has tested on sbm
        from layers import GraphConv
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.n_hid = n_hid

        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConv(n_feat,  n_hid))
        for _ in range(n_layers-1):
            self.gcs.append(GraphConv(n_hid,  n_hid))
        self.linear = nn.Linear(n_hid, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion

    def forward(self, x, adj):
        for ell in range(len(self.gcs)):
            x = self.gcs[ell](x, adj)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.linear(x)
        return x


#########################################################
#########################################################
#########################################################
# for mixhop
class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, dropout, criterion):
        super(MLP, self).__init__()
        self.conv1 = nn.Linear(n_feat, n_hid)
        self.conv2 = nn.Linear(n_hid, n_classes)
        self.criterion = criterion
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x)
        return x

class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, dropout, criterion):
        from torch_geometric.nn import GATConv
        super(GAT, self).__init__()
        n_head = 8
        self.conv1 = GATConv(n_feat, n_hid, heads=n_head, dropout=dropout)
        self.conv2 = GATConv(n_hid * n_head, n_classes, heads=1, dropout=dropout, concat=False)
        self.criterion = criterion
        self.dropout = dropout

    def forward(self, x, adj):
        adj_t = adj.to_dense()
        edge_index = adj_t.nonzero().t().contiguous().cuda() 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GAT2(nn.Module):
    def __init__(self, n_feat, n_hid, n_classes, dropout, criterion):
        from torch_geometric.nn import GATv2Conv
        super(GAT2, self).__init__()
        n_head = 8
        self.conv1 = GATv2Conv(n_feat, n_hid, heads=n_head, dropout=dropout)
        self.conv2 = GATv2Conv(n_hid * n_head, n_classes, heads=1, dropout=dropout, concat=False)
        self.criterion = criterion
        self.dropout = dropout

    def forward(self, x, adj):
        adj_t = adj.to_dense()
        edge_index = adj_t.nonzero().t().contiguous().cuda()  
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)   
        return x
