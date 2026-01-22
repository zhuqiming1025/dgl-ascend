"""
utilily module for LightGCN.
Provide various functions.
"""

import numpy as np
import torch

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(test_data,r,k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# The sampling function is implemented based on the DGL NGCF example (PyTorch backend).
def sample_triplets(g, n, etype=('user', 'interacts', 'item'), device='cpu'):
    # 1. All positive samples.
    users, pos_items = g.edges(etype=etype)
    num_edges = users.shape[0]
    num_items = g.num_nodes('item')
    # 2. Randomly sample n positive edges (if the number of edges is less than n, sample all)
    n = min(n, num_edges)
    perm = torch.randperm(num_edges, device=users.device)[:n]
    users = users[perm]
    pos_items = pos_items[perm]
    # 3.  For convenience in negative sampling, first get CSR format of Adjacency Matrix
    A = g.adj().csr()
    # 4. for every tuple (user, pos_item), find one negative_item.
    neg_items_list = []
    for user in users.tolist():
        while True:
            rand_neg_item = torch.randint(0, num_items, (1,)).item()
            start_col = A[0][user].item()
            end_col = A[0][user+1].item()
            if rand_neg_item not in A[1][start_col:end_col]:
                neg_items_list.append(rand_neg_item)
                break
    neg_items = torch.tensor(neg_items_list, device=users.device)
    # Return triplets (user, pos_item, neg_item)
    return users, pos_items, neg_items
