import torch
import torch.nn as nn
import torch.nn.functional as F
from gat import GAT
from GCNlayers import GraphConvolution

class Discriminator(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim3, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, 1),
        )

    def forward(self, x):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        z = self.dis(x)
        return z

class Graph_AE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim2 = 128,
                 gat_dropout=0,
                 multi_heads=2,
                 hidden_dim1=128):
        super(Graph_AE, self).__init__()
        self.gat = GAT(num_of_layers=2,
                       num_heads_per_layer=[multi_heads, multi_heads],
                       num_features_per_layer=[input_feat_dim, hidden_dim1, hidden_dim2],
                       dropout=gat_dropout)

        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, 0, act=nn.LeakyReLU(0.2, inplace=True))
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, 0, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, 0, act=lambda x: x)

        self.decode = InnerProductDecoder(0, act=lambda x: x)

    def encode_gat(self, in_nodes_features, edge_index):
        return self.gat((in_nodes_features, edge_index))

    def encode_gae(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, in_nodes_features, edge_index, encode=False, use_GAT=True):
        gae_info = None

        if use_GAT:
            hidden, out_nodes_features = self.encode_gat(in_nodes_features, edge_index)
        else:
            gae_info = self.encode_gae(in_nodes_features, edge_index)
            out_nodes_features = self.reparameterize(*gae_info)

        recon_graph = self.decode(out_nodes_features)
        return hidden, out_nodes_features, gae_info, recon_graph


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
