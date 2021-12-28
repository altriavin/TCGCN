import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data


def load_data(path, n_num, d_num):
    samples = load_samples(path)
    adj = construct_sparse_mx(np.array(samples), n_num, d_num)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    return samples, adj


def load_samples(path):
    samples = list()
    with open(path) as f:
        for line in f.readlines():
            line = line.strip('\n')
            tmp = line.split()
            samples.append([int(tmp[0]), int(tmp[1])])
    
    return samples


def construct_sparse_mx(points, n_num, d_num):
    src_points = np.array([[a, b+n_num] for a, b in points])
    trans_points = np.array([[b, a] for a, b in src_points])
    adj_points = np.concatenate((src_points, trans_points), axis=0)

    data = np.ones(len(adj_points))
    adj = sp.coo_matrix((data, (adj_points[:, 0], adj_points[:, 1])), shape=(n_num+d_num, n_num+d_num), dtype=np.float32)

    return adj


def normalize(adj):
    adj += sp.eye(adj.shape[0])
    d = np.array(adj.sum(1) + 1)
    d_half = sp.diags(np.power(d, -0.5).flatten())

    return d_half.dot(adj).dot(d_half)


def sparse_mx_to_torch_sparse_tensor(adj):
    adj = adj.tocoo().astype(np.float32)
    indices = torch.cuda.LongTensor(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.cuda.FloatTensor(adj.data)
    shape = torch.Size(adj.shape)

    return torch.sparse.FloatTensor(indices, values, shape)
