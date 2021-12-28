import os
import random
import numpy as np
import pandas as pd


def k_fold_split(index_df, k, suffix):
    k_fold = []
    index = set(range(index_df.shape[0]))
    for i in range(k):
        tmp = random.sample(list(index), int(1.0 / k * index_df.shape[0]))
        k_fold.append(tmp)
        index -= set(tmp)


    if len(index)!= 0:
        picked = np.arange(k)
        np.random.shuffle(picked)
        picked = picked[:len(index)]
        for n, i in enumerate(index):
            k_fold[picked[n]].append(i)

    data_path_base = '../data/samples/'
    if not (os.path.exists(data_path_base)):
        os.makedirs(data_path_base)
    for i in range(k):
        print('Fold-%d........' % (i + 1))
        tra = []
        dev = k_fold[i]
        for j in range(k):
            if i != j:
                tra += k_fold[j]
        train_samples = index_df.iloc[tra].to_numpy()
        test_samples = index_df.iloc[dev].to_numpy()
        write_samples(train_samples, data_path_base + 'train_' + suffix + '_%d.txt' % i)
        write_samples(test_samples, data_path_base + 'test_' + suffix + '_%d.txt' % i)
    print('done!')


def write_samples(samples, path):
    with open(path, 'w') as f:
        for i in samples:
            assert len(i) == 2, 'length doesn\'t match while writing samples'
            f.write(str(i[0])+'\t'+str(i[1])+'\n')


if __name__ == '__main__':
    pos_df = pd.read_table('../data/ncrna_mesh_pos_index.txt', header=None)
    neg_df = pd.read_table('../data/ncrna_mesh_neg_index.txt', header=None)
    k_fold_split(pos_df, 5, 'pos')
    k_fold_split(neg_df, 5, 'neg')
