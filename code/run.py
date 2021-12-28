import os
from random import sample
import time
import json
import argparse
from argparse import Namespace
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import LRGCPND
from utils import load_data, load_samples
from metrics import evaluate
from sklearn.metrics import auc


os.environ['CUDA_VISIBLE_DEVICES'] =','.join(map(str, [0]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_num', type=int, default=2319, help='Number of ncRNAs.')
    parser.add_argument('--d_num', type=int, default=556, help='Number of drugs.')
    parser.add_argument('-K', type=int, default=4, help='Depth of layers.')
    parser.add_argument('-S', type=int, default=32, help='Embedding size.')
    parser.add_argument('-r', '--reg', type=float, default=0.05, help='Coefficient of L2 regularization.')
    parser.add_argument('-l', '--lr', type=float, default=0.003, help='Initial learning rate.')
    parser.add_argument('-e', '--epochs', type=int, default=40, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size to train.')
    parser.add_argument('-f', '--fold', type=int, default=5, help='Number of folds for cross validation.')
    parser.add_argument('-t', '--time', type=str, default=None, help='Timestamp in milliseconds for training.')
    parser.add_argument('--save_models', action='store_true', default=False, help='Save trained models.')
    parser.add_argument('--have_trained', action='store_true', default=False, help='Have trained models.')
    parser.add_argument('--save_details', action='store_true', default=False, help='Save test details.')
    parser.add_argument('--save_special', action='store_true', default=False, help='Save special details.')
    args = parser.parse_args()
    return args


def train(data_pos_path, data_neg_path, n_num, d_num, K, E_size, reg, lr, epochs, batch_size):
    train_pos_samples, adj = load_data(data_pos_path, n_num, d_num)
    train_neg_samples = load_samples(data_neg_path)
    samples = train_pos_samples + train_neg_samples
    labels = [1]*len(train_pos_samples) + [0]*len(train_neg_samples)
    train_dataset = TensorDataset(torch.LongTensor(samples), torch.FloatTensor(labels))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)

    model = LRGCPND(n_num, d_num, adj, K, E_size, reg)
    model=model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for e in range(epochs):
        model.train()
        loss_sum = []
        start_time = time.time()

        for data in train_loader:
            inps, tgts = data
            inps = inps.cuda()
            tgts = tgts.cuda()
            n = inps[:, 0]
            d_i = inps[:, 1]

            optimizer.zero_grad()
            pre_i, l2_loss = model(n, d_i)
            loss = criterion(pre_i, tgts) + l2_loss
            loss.backward()
            optimizer.step()

            loss_sum.append(loss.item())

        train_loss=round(np.mean(loss_sum),4)
        end_time = time.time()
        str_e = 'epoch:%-3d\ttime:%.2f\ttrain loss:%.4f' % (e, end_time-start_time, train_loss)
        print(str_e)
    return model


def test(model, data_pos_path, data_neg_path, batch_size, threshold, id=309):
    model.eval()
    test_pos_samples = load_samples(data_pos_path)
    test_neg_samples = load_samples(data_neg_path)
    all_pos_spe_samples = load_samples('../data/ncrna_mesh_pos_index.txt')
    all_neg_spe_samples = load_samples('../data/ncrna_mesh_neg_index.txt')

    test_pos_arr  = np.array(test_pos_samples)
    test_neg_arr = np.array(test_neg_samples)
    test_pos_spe = np.where(test_pos_arr[:, 1]==id)[0]
    test_neg_spe = np.where(test_neg_arr[:, 1]==id)[0]
    ng_num = len(test_pos_spe) - len(test_neg_spe)

    if ng_num>=0:
        for t in range(ng_num):
            n=np.random.randint(2319)
            tmp = [n, id]
            while tmp in all_pos_spe_samples or tmp in all_neg_spe_samples:
                n=np.random.randint(2319)
                tmp = [n, id]
            test_neg_samples.append(tmp)

    samples = test_pos_samples + test_neg_samples
    labels = [1]*len(test_pos_samples) + [0]*len(test_neg_samples)
    test_dataset = TensorDataset(torch.LongTensor(samples), torch.FloatTensor(labels))
    test_loader = DataLoader(test_dataset, batch_size)
    results, fpr, tpr, spe_results, spe_fpr, spe_tpr, _ = evaluate(model, test_loader, threshold, id)

    return results, fpr, tpr, spe_results, spe_fpr, spe_tpr


def run(args):
    data_path_base = '../data/samples/'
    model_path_base = '../model'
    result_path_base = '../result'
    str_time = str(round(time.time()*1000))

    if args.save_models:
        model_path_base += '/' + str_time
        os.makedirs(model_path_base)
    if not (os.path.exists(result_path_base)):
        os.makedirs(result_path_base)

    avg_res = defaultdict(int)
    spe_avg_res = defaultdict(int)
    print('-'*20 + 'start' + '-'*20)
    print('time: ' + str_time)
    print(vars(args))
    for i in range(args.fold):
        train_pos_path = data_path_base + 'train_pos_' + str(i) + '.txt'
        train_neg_path = data_path_base + 'train_neg_' + str(i) + '.txt'
        test_pos_path = data_path_base + 'test_pos_' + str(i) + '.txt'
        test_neg_path = data_path_base + 'test_neg_' + str(i) + '.txt'
        print('-'*20 + ('fold-%d-start' % i) + '-'*20)
        model = train(train_pos_path, train_neg_path, args.n_num, args.d_num, args.K, args.S, args.reg, args.lr, args.epochs, args.batch)

        if args.save_models:
            torch.save(model.state_dict(), model_path_base + '/' + str(i) + '.pt')
        
        print('-'*20 + ('fold-%d-end' % i) + '-'*20)
        results, _, _, spe_results, _, _ = test(model, test_pos_path, test_neg_path, 32, 0.5)
        for k, v in results.items():
            print('%s\t%.4f' % (k, v))
            avg_res[k] += v
        print('-'*10 + 'SPE' + '-'*10)
        for k, v in spe_results.items():
            print('%s\t%.4f' % (k, v))
            spe_avg_res[k] += v
    
    print('-'*20 + 'end' + '-'*20)
    print('time: ' + str_time)
    print_average_results(args, avg_res)
    for k, v in spe_avg_res.items():
        spe_avg_res[k] = round(v / 5, 4)
    print('-'*10 + 'SPE_AVG' + '-'*10)
    print(spe_avg_res)

    res = dict()
    res['args'] = str(args)
    res['performance'] = avg_res
    with open(result_path_base + '/' + str_time + '.json', 'w') as f:
        f.write(json.dumps(res))


def load_args(str_time):
    assert str_time, 'illegal timestamp'

    result_path = '../result/' + str_time + '.json'
    with open(result_path, 'r') as f:
        res = json.loads(f.read())
        args = eval(res['args'])
    return args


def load_and_test(str_time, save_details=False, save_special=False):
    data_path_base = '../data/samples/'
    model_path_base = '../model'

    args = load_args(str_time)
    args.have_trained = True
    args.time = str_time
    avg_res = defaultdict(int)
    spe_avg_res = defaultdict(int)

    fpr_list = []
    tpr_list = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    auc_list = []
    aupr_list = []
    acc_list = []
    p_list = []
    r_list = []
    f1_list = []

    spe_fpr_list = []
    spe_tpr_list = []
    spe_tprs = []
    spe_mean_fpr = np.linspace(0, 1, 100)
    spe_auc_list = []
    spe_aupr_list = []
    spe_acc_list = []
    spe_p_list = []
    spe_r_list = []
    spe_f1_list = []

    for i in range(args.fold):
        train_pos_path = data_path_base + 'train_pos_' + str(i) + '.txt'
        test_pos_path = data_path_base + 'test_pos_' + str(i) + '.txt'
        test_neg_path = data_path_base + 'test_neg_' + str(i) + '.txt'

        _, adj = load_data(train_pos_path, args.n_num, args.d_num)
        model = LRGCPND(args.n_num, args.d_num, adj, args.K, args.S, args.reg)
        model=model.to('cuda')

        model_path = model_path_base + '/' + str_time + '/' + str(i) + '.pt'
        model.load_state_dict(torch.load(model_path))

        print('-'*20 + ('fold-%d' % i) + '-'*20)
        results, fpr, tpr, spe_results, spe_fpr, spe_tpr = test(model, test_pos_path, test_neg_path, 32, 0.5)
        for k, v in results.items():
            print('%s\t%.4f' % (k, v))
            avg_res[k] += v
        print('-'*10 + 'SPE' + '-'*10)
        for k, v in spe_results.items():
            print('%s\t%.4f' % (k, v))
            spe_avg_res[k] += v

        if save_details:
            fpr_list, tpr_list, tprs, auc_list, aupr_list, acc_list, p_list, r_list, f1_list = append_results(fpr_list, tpr_list, tprs, mean_fpr, auc_list, aupr_list, acc_list, p_list, r_list, f1_list, results, fpr, tpr)
        
        if save_special:
            spe_fpr_list, spe_tpr_list, spe_tprs, spe_auc_list, spe_aupr_list, spe_acc_list, spe_p_list, spe_r_list, spe_f1_list = append_results(spe_fpr_list, spe_tpr_list, spe_tprs, spe_mean_fpr, spe_auc_list, spe_aupr_list, spe_acc_list, spe_p_list, spe_r_list, spe_f1_list, spe_results, spe_fpr, spe_tpr)

    if save_details:
        save_results(str_time, fpr_list, tpr_list, tprs, mean_fpr, auc_list, aupr_list, acc_list, p_list, r_list, f1_list, 'ncrna_mesh')
    
    if save_special:
        save_results(str_time, spe_fpr_list, spe_tpr_list, spe_tprs, spe_mean_fpr, spe_auc_list, spe_aupr_list, spe_acc_list, spe_p_list, spe_r_list, spe_f1_list, 'ncrna_mesh_special')

    print_average_results(args, avg_res)
    for k, v in spe_avg_res.items():
        spe_avg_res[k] = round(v / 5, 4)
    print('-'*10 + 'SPE_AVG' + '-'*10)
    print(spe_avg_res)


def load_and_test_special(str_time, id, threshold=0.5, save_special=False):
    df = pd.read_csv('../data/ncrna_mesh.csv', index_col=0)
    prefix = id.replace(':', '_')
    id = df.columns.get_loc(id)

    data_path_base = '../data/samples/'
    model_path_base = '../model'

    args = load_args(str_time)
    args.have_trained = True
    args.time = str_time
    avg_res = defaultdict(int)
    spe_avg_res = defaultdict(int)

    spe_fpr_list = []
    spe_tpr_list = []
    spe_tprs = []
    spe_mean_fpr = np.linspace(0, 1, 100)
    spe_auc_list = []
    spe_aupr_list = []
    spe_acc_list = []
    spe_p_list = []
    spe_r_list = []
    spe_f1_list = []

    for i in range(args.fold):
        train_pos_path = data_path_base + 'train_pos_' + str(i) + '.txt'
        test_pos_path = data_path_base + 'test_pos_' + str(i) + '.txt'
        test_neg_path = data_path_base + 'test_neg_' + str(i) + '.txt'

        _, adj = load_data(train_pos_path, args.n_num, args.d_num)
        model = LRGCPND(args.n_num, args.d_num, adj, args.K, args.S, args.reg)
        model=model.to('cuda')

        model_path = model_path_base + '/' + str_time + '/' + str(i) + '.pt'
        model.load_state_dict(torch.load(model_path))

        print('-'*20 + ('fold-%d' % i) + '-'*20)
        results, _, _, spe_results, spe_fpr, spe_tpr = test(model, test_pos_path, test_neg_path, 32, threshold, id)
        for k, v in results.items():
            print('%s\t%.4f' % (k, v))
            avg_res[k] += v
        print('-'*10 + 'SPE' + '-'*10)
        for k, v in spe_results.items():
            print('%s\t%.4f' % (k, v))
            spe_avg_res[k] += v
        
        if save_special:
            spe_fpr_list, spe_tpr_list, spe_tprs, spe_auc_list, spe_aupr_list, spe_acc_list, spe_p_list, spe_r_list, spe_f1_list = append_results(spe_fpr_list, spe_tpr_list, spe_tprs, spe_mean_fpr, spe_auc_list, spe_aupr_list, spe_acc_list, spe_p_list, spe_r_list, spe_f1_list, spe_results, spe_fpr, spe_tpr)
    
    if save_special:
        save_results(str_time, spe_fpr_list, spe_tpr_list, spe_tprs, spe_mean_fpr, spe_auc_list, spe_aupr_list, spe_acc_list, spe_p_list, spe_r_list, spe_f1_list, prefix, False)

    print_average_results(args, avg_res)
    for k, v in spe_avg_res.items():
        spe_avg_res[k] = round(v / 5, 4)
    print('-'*10 + 'SPE_AVG' + '-'*10)
    print(spe_avg_res)


def case_study(str_time, id, top_k=100, save=False):
    df = pd.read_csv('../data/ncrna_mesh.csv', index_col=0)
    arr = df.loc[:, id].to_numpy()
    # screen unknown associations
    arr = np.where(arr==0)[0]
    d = df.columns.get_loc(id)

    samples = [[n, d] for n in arr]
    case_dataset = TensorDataset(torch.LongTensor(samples))
    case_loader = DataLoader(case_dataset, 32)
    case_dict = dict()

    args = load_args(str_time)
    args.have_trained = True
    args.time = str_time

    data_pos_path = '../data/ncrna_mesh_pos_index.txt'
    model_path = '../model/' + str_time + '/0.pt'
    result_path_base = '../result/' + str_time
    if not (os.path.exists(result_path_base)):
        os.makedirs(result_path_base)

    _, adj = load_data(data_pos_path, args.n_num, args.d_num)
    model = LRGCPND(args.n_num, args.d_num, adj, args.K, args.S, args.reg)
    model=model.to('cuda')

    model.load_state_dict(torch.load(model_path))

    for data in case_loader:
        data = data[0].cuda()
        n = data[:, 0]
        d_i = data[:, 1]

        prediction_i, _ = model(n, d_i)

        for e in range(len(n)):
            case_dict[n[e].item()] = prediction_i[e].item()
          
    case_sorted = sorted(case_dict.items(), key=lambda x:x[1], reverse=True)
    # case_sorted = case_sorted[:top_k]
    print(case_sorted)
    # result_path = result_path_base + '/top_' + str(top_k) + '.txt'
    result_path = result_path_base + '/top.txt'
    if save:
        with open(result_path, 'w') as f:
            for i in case_sorted:
                assert len(i) == 2, 'length doesn\'t match while writing samples'
                d_i = df.index[i[0]]
                f.write(d_i+'\n')


def print_average_results(args, avg_res): 
    print('-'*20 + 'Args' + '-'*20)
    print(vars(args))
    for k, v in avg_res.items():
        avg_res[k] = round(v / 5, 4)
    print('-'*20 + 'AVG' + '-'*20)
    print(avg_res)


def append_results(fpr_list, tpr_list, tprs, mean_fpr, auc_list, aupr_list, acc_list, p_list, r_list, f1_list, results, fpr, tpr):
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    auc_list.append(results['AUC'])
    aupr_list.append(results['AUPR'])
    acc_list.append(results['ACC.'])
    p_list.append(results['P.'])
    r_list.append(results['R.'])
    f1_list.append(results['F1'])

    return fpr_list, tpr_list, tprs, auc_list, aupr_list, acc_list, p_list, r_list, f1_list


def save_results(str_time, fpr_list, tpr_list, tprs, mean_fpr, auc_list, aupr_list, acc_list, p_list, r_list, f1_list, prefix, save_details=True):
    detail_path_base = '../result/' + str_time
    if not (os.path.exists(detail_path_base)):
        os.makedirs(detail_path_base)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)
    auc_list = np.array(auc_list)
    mean_fpr = np.array(mean_fpr)
    mean_tpr = np.array(mean_tpr)

    if save_details:
        np.save(detail_path_base + '/' + prefix + '_fpr.npy', fpr_list)
        np.save(detail_path_base + '/' + prefix + '_tpr.npy', tpr_list)
        np.save(detail_path_base + '/' + prefix + '_AUC.npy', auc_list)
        np.save(detail_path_base + '/' + prefix + '_AUPR.npy', aupr_list)
        np.save(detail_path_base + '/' + prefix + '_mean_fpr_tpr_auc.npy', [mean_fpr, mean_tpr, mean_auc])
        np.save(detail_path_base + '/' + prefix + '_acc.npy', acc_list)
        np.save(detail_path_base + '/' + prefix + '_p.npy', p_list)
        np.save(detail_path_base + '/' + prefix + '_r.npy', r_list)
        np.save(detail_path_base + '/' + prefix + '_f1.npy', f1_list)
    else:
        np.save(detail_path_base + '/' + prefix + "_mean_fpr_tpr.npy", [mean_fpr, mean_tpr])
        np.save(detail_path_base + '/' + prefix + "_mean_metrics.npy", [np.mean(auc_list), np.mean(aupr_list), np.mean(acc_list), np.mean(p_list), np.mean(r_list), np.mean(f1_list)])


if __name__ == '__main__':
    args = get_args()
    if args.have_trained:
        load_and_test(args.time, args.save_details, args.save_special)
    else:
        run(args)
