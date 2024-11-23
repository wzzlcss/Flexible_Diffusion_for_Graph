import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax
from critical_look_utils.modules import FeedForwardModule, _check_dim_and_num_heads_consistency

class DGATModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.edge_linear = nn.Linear(in_features=2, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_edges = nn.Linear(in_features=dim, out_features=num_heads)

        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)
        edge_attr = self.edge_linear(graph.edata['edge_feat'])

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores_e = self.attn_edges(edge_attr)

        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = attn_scores + attn_scores_e

        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        x = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)

        x = self.feed_forward_module(graph, x)

        return x

class DGATSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, dropout, **kwargs):
        super().__init__()

        _check_dim_and_num_heads_consistency(dim, num_heads)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.edge_linear = nn.Linear(in_features=2, out_features=dim)

        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_edges = nn.Linear(in_features=dim, out_features=num_heads)

        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)

    def forward(self, graph, x):
        x = self.input_linear(x)
        edge_attr = self.edge_linear(graph.edata['edge_feat'])

        attn_scores_u = self.attn_linear_u(x)
        attn_scores_v = self.attn_linear_v(x)
        attn_scores_e = self.attn_edges(edge_attr)

        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = attn_scores + attn_scores_e

        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        x = x.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, x, attn_probs)
        x = x.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        x = torch.cat([x, message], axis=1)

        x = self.feed_forward_module(graph, x)

        return x

