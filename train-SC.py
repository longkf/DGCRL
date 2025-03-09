import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from torch.optim import Adam
from torch_geometric import seed_everything
from tqdm import trange
from load_SC import DataSet_SC
from model import DGCRL, Decoder
from scheduler import CosineDecayScheduler
from transforms import get_graph_drop_transform
from utils import set_random_zeros_by_proportion


def train(step):
    model.train()
    lr = lr_scheduler.get(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    mm = 1 - mm_scheduler.get(step)
    data1, data2 = transform_1(train_data), transform_2(train_data)
    Q, K = model(data1, data2)
    alignment_loss = 1 - F.cosine_similarity(Q, K.detach(), dim=-1).mean()
    idx = torch.randperm(data.num_nodes)
    U = Q[idx]
    separation_loss = F.cosine_similarity(Q, U, dim=-1).mean()
    loss = alignment_loss + separation_loss * args.lam
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_target_network(mm)
    return loss.item()


def link_train():
    decoder.train()
    pred = decoder(Q, K, train_data.edge_label_index).squeeze()
    loss = criterion(pred, train_data.edge_label)
    optimizer_link.zero_grad()
    loss.backward()
    optimizer_link.step()
    return loss.item()


def link_val():
    decoder.eval()
    pred = decoder(Q, K, val_data.edge_label_index).squeeze().detach().cpu().numpy()
    roc = roc_auc_score(val_data.edge_label.cpu().numpy(), pred)
    ap = average_precision_score(val_data.edge_label.cpu().numpy(), pred)
    return roc, ap


def link_test():
    decoder.eval()
    pred = decoder(Q, K, test_data.edge_label_index).squeeze().detach().cpu().numpy()
    roc = roc_auc_score(test_data.edge_label.cpu().numpy(), pred)
    ap = average_precision_score(test_data.edge_label.cpu().numpy(), pred)
    fpr, tpr, _ = roc_curve(test_data.edge_label.cpu().numpy(), pred, pos_label=1)
    precision, recall, _ = precision_recall_curve(test_data.edge_label.cpu().numpy(), pred, pos_label=1)
    return roc, ap, fpr, tpr, precision, recall


if __name__ == '__main__':
    torch.set_printoptions(profile="full", sci_mode=False)
    seed_everything(123)
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr_warmup_epochs', type=int, default=1000)
    # parser.add_argument('--data_path', type=str, default='./data/dream4/100gene')
    # parser.add_argument('--data_path', type=str, default='./data/dream5')
    parser.add_argument('--data_path', type=str, default='./data/single-cell')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--out_channels', type=int, default=16)
    parser.add_argument('--decoder_hidden_channels', type=int, default=64)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--lr1', type=float, default=0.01)
    parser.add_argument('--lr2', type=float, default=0.01)
    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--mm', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--drop_edge_p_1', type=float, default=0.1)
    parser.add_argument('--drop_feat_p_1', type=float, default=0.1)
    parser.add_argument('--reverse_edge_1', type=bool, default=False)
    parser.add_argument('--drop_edge_p_2', type=float, default=0.1)
    parser.add_argument('--drop_feat_p_2', type=float, default=0.1)
    parser.add_argument('--reverse_edge_2', type=bool, default=True)
    parser.add_argument('--link_epochs', type=int, default=1000)
    args = parser.parse_args()
    rocs = []
    aps = []

    for run in range(args.runs):
        dataset = DataSet_SC(args.data_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = dataset[0]
        print(data)
        data.x = set_random_zeros_by_proportion(data.x, 0)
        if torch.cuda.is_available():
            data = data.cuda()
        split = T.RandomLinkSplit(
            num_val=0.2,
            num_test=0.2,
            is_undirected=False,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
        )

        train_data, val_data, test_data = split(data)

        transform_1 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_1, drop_feat_p=args.drop_feat_p_1,
                                               do_reverse=args.reverse_edge_1)
        transform_2 = get_graph_drop_transform(drop_edge_p=args.drop_edge_p_2, drop_feat_p=args.drop_feat_p_2,
                                               do_reverse=args.reverse_edge_2)
        model = DGCRL(data.x.shape[1], args.hidden_channels, args.out_channels, batch_norm=args.batch_norm).to(device)
        # print(model)
        optimizer = Adam(model.trainable_parameters(), lr=args.lr1, weight_decay=args.weight_decay)
        lr_scheduler = CosineDecayScheduler(args.lr1, args.lr_warmup_epochs, args.epochs)
        mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epochs)

        for epoch in trange(1, 1 + args.epochs):
            train(epoch - 1)
        decoder = Decoder(args.out_channels, args.decoder_hidden_channels, 1).to(device)
        data_q = copy.deepcopy(train_data)
        data_k = copy.deepcopy(data_q)
        data_k.edge_index = torch.flip(data_k.edge_index, dims=[0])
        encoder = copy.deepcopy(model)
        Q, K = encoder(data_q, data_k)
        Q = nn.Embedding.from_pretrained(Q, freeze=True)
        K = nn.Embedding.from_pretrained(K, freeze=True)

        test_data = test_data.cpu()
        Q = Q.cpu()
        K = K.cpu()
        src = Q(test_data.edge_label_index[0])
        dst = K(test_data.edge_label_index[1])
        test_data = test_data.to(device)
        Q = Q.to(device)
        K = K.to(device)

        optimizer_link = Adam(decoder.parameters(), lr=args.lr2, weight_decay=args.weight_decay)
        criterion = torch.nn.BCELoss().to(device)
        best_roc = 0
        for epoch in range(1, 1 + args.link_epochs):
            loss = link_train()
            roc, ap = link_val()
            if roc >= best_roc:
                best_roc = roc
                torch.save(decoder, args.save_path + 'model.pt')
                print('*', end='')
            print(f'epoch: {epoch} loss: {loss} roc: {roc} ap: {ap}')
        decoder = torch.load(args.save_path + 'model.pt')
        roc, ap, fpr, tpr, precision, recall = link_test()
        print(f'roc: {roc} ap: {ap}')
        rocs.append(roc)
        aps.append(ap)
        # plt.plot(fpr, tpr, 'b', label='ROC (area = {0:.2f})'.format(roc), lw=2)
        # plt.plot([0, 1], [0, 1], 'r--')
        # plt.legend(loc="lower right")
        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.show()
    ret = [np.mean(rocs), np.std(rocs), np.mean(aps), np.std(aps)]
    print(ret)
