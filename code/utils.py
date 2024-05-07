import os
import torch
import numpy as np
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryMatthewsCorrCoef, BinaryAveragePrecision,BinaryAUROC
from NTXentLoss import NTXentLoss

os.environ['PATH'] += ':/mnt/wzw/anaconda3/envs/parapred-pytorch/bin/'
device = torch.device("cuda")


def get_k_fold_data(K, i, X):
    assert K > 1
    fold_size = len(X) // K

    X_train, X_val, X_test = None, None, None

    tmp_list = list(range(K))
    idx_i = tmp_list.index(i)
    del tmp_list[idx_i]
    v = tmp_list[-1]

    for j in range(K):
        idx = slice(j * fold_size, (j + 1) * fold_size)

        X_part = X[idx]
        if j == i:
            X_test = X_part
        elif j == v:
            X_val = X_part
        elif X_train is None:
            X_train = X_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
    return X_train, X_val, X_test

def evalution_prot(preds, targets):
    # AUROC
    auroc = BinaryAUROC().to(device)
    auroc.update(preds, targets)
    auroc_i = (auroc.compute()).item()
    # AUPRC
    auprc = BinaryAveragePrecision().to(device)
    auprc.update(preds, targets)
    auprc_i = (auprc.compute()).item()
    # Precision
    precision = BinaryPrecision().to(device)
    precision_i = precision(preds, targets)
    # Recall
    recall = BinaryRecall().to(device)
    recall_i = recall(preds, targets)
    # MCC
    mcc = BinaryMatthewsCorrCoef().to(device)
    mcc_i = mcc(preds, targets)

    return auprc_i, auroc_i, mcc_i.item()

def consine_inter(A, B):
    dot_product = torch.sum(A * B, dim=1)
    norm_A = torch.norm(A, dim=1)
    norm_B = torch.norm(B, dim=1)
    cosine_similarity = dot_product / ((norm_A * norm_B) + 1e-8)
    return (cosine_similarity)

# dis
def dis_pairs(coord_1, coord_2):
    coord_1_x = coord_1[-3]
    coord_1_y = coord_1[-2]
    coord_1_z = coord_1[-1]
    coord_2_x = coord_2[-3]
    coord_2_y = coord_2[-2]
    coord_2_z = coord_2[-1]
    distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + (float(coord_1_y) - float(coord_2_y)) ** 2 + (
                float(coord_1_z) - float(coord_2_z)) ** 2)
    return distance

# index_ink
def index_mink(data, k):
    Lst = data[:]
    index_k = []
    for i in range(k):
        index_i = Lst.index(min(Lst))
        index_k.append(index_i)
        Lst[index_i] = float('inf')
    return (index_k)

def CreateGearnetGraph(data):
    from torchdrug import data as drugdata
    # AG
    edge_AG_radius = (np.array(data["edge_AG"] + ([[1] * len(data["edge_AG"][0])])).T).tolist()
    num_nodes_AG = max(max(np.array(edge_AG_radius)[:, 0]), max(np.array(edge_AG_radius)[:, 1])) + 1
    edge_AG_seq = []
    for p in range(num_nodes_AG - 1):
        edge_AG_seq.append([p, p + 1, 0])
    edge_AG_10nearest = []
    for p in range(num_nodes_AG):
        dis_pq = []
        for q in range(num_nodes_AG):
            dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
        near10_q = index_mink(dis_pq, 11)
        del near10_q[near10_q.index(p)]
        near10_AG_p = list(map(lambda x: [p, x, 2], near10_q))
        edge_AG_10nearest = edge_AG_10nearest + near10_AG_p
    edge_AG = edge_AG_seq + edge_AG_radius + edge_AG_10nearest
    graph_AG = drugdata.Graph(edge_AG, num_node=num_nodes_AG, num_relation=3).to(device)
    node_embedding_AG = torch.tensor(data["vertex_AG"], dtype=torch.float).to(device)
    ag_edge_ind = [graph_AG, node_embedding_AG]
    # AB
    edge_AB_radius = (np.array(data["edge_AB"] + ([[1] * len(data["edge_AB"][0])])).T).tolist()
    num_nodes_AB = max(max(np.array(edge_AB_radius)[:, 0]), max(np.array(edge_AB_radius)[:, 1])) + 1
    edge_AB_seq = []
    for p in range(num_nodes_AG - 1):
        edge_AG_seq.append([p, p + 1, 0])
    edge_AB_10nearest = []
    for p in range(num_nodes_AB):
        dis_pq = []
        for q in range(num_nodes_AB):
            dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
        near10_q = index_mink(dis_pq, 11)
        del near10_q[near10_q.index(p)]
        near10_AB_p = list(map(lambda x: [p, x, 2], near10_q))
        edge_AB_10nearest = edge_AB_10nearest + near10_AB_p
    edge_AB = torch.tensor(edge_AB_seq + edge_AB_radius + edge_AB_10nearest)
    graph_AB = drugdata.Graph(edge_AB, num_node=num_nodes_AB, num_relation=3).to(device)
    node_embedding_AB = torch.tensor(data["vertex_AB"], dtype=torch.float).to(device)
    ab_edge_ind = [graph_AB, node_embedding_AB]

    return ag_edge_ind, ab_edge_ind

# k-nearest
def CreateKnearestEdge(data):
    # AG
    num_nodes_AG = data["vertex_AB"].shape[0]
    edge_AG_10nearest_p = []
    edge_AG_10nearest_q = []
    for p in range(num_nodes_AG):
        dis_pq = []
        for q in range(num_nodes_AG):
            dis_pq.append(dis_pairs(data["coord_AG"][p], data["coord_AG"][q]))
        near10_q = index_mink(dis_pq, 11)
        del near10_q[near10_q.index(p)]
        near10_p = [p] * 10
        edge_AG_10nearest_p = edge_AG_10nearest_p + near10_p
        edge_AG_10nearest_q = edge_AG_10nearest_q + near10_q
    edge_AG_10nearest = [edge_AG_10nearest_p, edge_AG_10nearest_q]
    # AB
    num_nodes_AB = data["vertex_AB"].shape[0]
    edge_AB_10nearest_p = []
    edge_AB_10nearest_q = []
    for p in range(num_nodes_AB):
        dis_pq = []
        for q in range(num_nodes_AB):
            dis_pq.append(dis_pairs(data["coord_AB"][p], data["coord_AB"][q]))
        near10_q = index_mink(dis_pq, 11)
        del near10_q[near10_q.index(p)]
        near10_p = [p] * 10
        edge_AB_10nearest_p = edge_AB_10nearest_p + near10_p
        edge_AB_10nearest_q = edge_AB_10nearest_q + near10_q
    edge_AB_10nearest = [edge_AB_10nearest_p, edge_AB_10nearest_q]
    return edge_AG_10nearest, edge_AB_10nearest