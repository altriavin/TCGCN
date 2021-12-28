import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
    

    def forward(self, input, adj):
        output = torch.sparse.mm(adj, input)
        return output


class LRGCPND(nn.Module):
    def __init__(self, n_num, d_num, adj, K, E_size, reg):
        super(LRGCPND, self).__init__()
        self.adj = adj
        self.n_num = n_num
        self.d_num = d_num
        self.E = nn.Embedding(n_num + d_num, E_size)
        self.reg = reg
        self.convs = nn.ModuleList([GraphConvolution(E_size, E_size) for i in range(K)])

        stdv = 1. / math.sqrt(self.E.weight.size(1))
        nn.init.normal_(self.E.weight, std=stdv)
    

    def forward(self, n, d_i):
        embedding = self.E.weight
        l2_reg = torch.cuda.FloatTensor([0])
        for i, conv in enumerate(self.convs):
            if i:
                x = conv(x, self.adj)
                gcn_embedding = torch.add(gcn_embedding, x)
            else:
                x = conv(embedding, self.adj)
                gcn_embedding = torch.add(embedding, x)

        
        n = F.embedding(n, gcn_embedding)
        d_i = F.embedding(self.n_num + d_i, gcn_embedding)

        pre_i = torch.sigmoid((n * d_i).sum(dim=-1))
        
        l2_reg += (n**2+d_i**2).sum(dim=-1).mean()
        l2_loss = self.reg * l2_reg

        return pre_i, l2_loss
