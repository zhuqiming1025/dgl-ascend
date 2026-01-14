import dgl
import  sys
from parser import parse_args
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
import argparse



class DGLDataset:
    def __init__(self, path):
        self.path = path
        self._load_data()
        self._build_graph()

    def _load_data(self):
        train_user, train_item = [], []

        with open(self.path + "/train.txt") as f:
            for line in f:
                l = line.strip().split()
                u = int(l[0])
                items = [int(i) for i in l[1:]]
                train_user.extend([u] * len(items))
                train_item.extend(items)

        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)

        self.n_users = self.train_user.max() + 1
        self.n_items = self.train_item.max() + 1
    def _build_graph(self):
        # user → item
        src = torch.from_numpy(self.train_user)
        dst = torch.from_numpy(self.train_item)

        # 暂时保存异构图
        self.hetero_graph = dgl.heterograph({
            ('user', 'interacts', 'item'): (src, dst),
        })

        # Create bidirectional bipartite graph
        # User nodes：0 to n_users-1
        # Item nodes：n_users to n_users+n_items-1
        dst_offset = dst + self.n_users
        
        # 构建双向边：user->item 和 item->user
        src_bidir = torch.cat([src, dst_offset])
        dst_bidir = torch.cat([dst_offset, src])
        self.graph = dgl.graph((src_bidir, dst_bidir), num_nodes=self.n_users + self.n_items)

        print("Hetero graph:", self.hetero_graph)
        print("Bipartite graph:", self.graph)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_size, n_layers):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                dglnn.GraphConv(emb_size, emb_size, weight=False, bias=False)
            )
        # Initialize user and item node embeddings with a normal (Gaussian) distribution.
        self.embedding_user = nn.Embedding(n_users, emb_size)
        self.embedding_item = nn.Embedding(n_items, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding_user.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.embedding_item.weight, mean=0.0, std=0.1)

    def forward(self, g):
        h_user = self.embedding_user.weight
        h_item = self.embedding_item.weight
        
        # Store embeddings from all layers
        embeddingList_user = [h_user]
        embeddingList_item = [h_item]
        
        for i, layer in enumerate(self.layers):
            
            h_all = torch.cat([h_user, h_item], dim=0)
            h_all_new = layer(g, h_all)
            h_user = h_all_new[:self.n_users]
            h_item = h_all_new[self.n_users:]
            embeddingList_user.append(h_user)
            embeddingList_item.append(h_item)
        # Take the embedding of each layer and compute the average (the core idea of LightGCN)
        final_user_emb = torch.mean(torch.stack(embeddingList_user, dim=1), dim=1)
        final_item_emb = torch.mean(torch.stack(embeddingList_item, dim=1), dim=1)
        
        # Concatenate user and item embeddings, and return the complete embedding
        all_emb = torch.cat([final_user_emb, final_item_emb], dim=0)
        return all_emb


class BPRLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.userEmb = model.embedding_user.weight
        self.itemEmb = model.embedding_item.weight
        self.all_embedding = torch.cat([self.userEmb, self.itemEmb], dim=0)
    
    def forward(self, user_emb, pos_emb, neg_emb, batch_user, batch_pos, batch_neg):
        # Compute the BPRLoss and the reg loss
        pos_score = torch.mul(user_emb, pos_emb)
        pos_score = torch.sum(pos_score, dim=1)
        neg_score = torch.mul(user_emb, neg_emb)
        neg_score = torch.sum(neg_score, dim=1)
        loss = torch.mean(F.softplus(neg_score - pos_score))
        userEmb0 = self.all_embedding[batch_user]
        posEmb0 = self.all_embedding[batch_pos]
        negEmb0 = self.all_embedding[batch_neg]
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(batch_user))
        return loss, reg_loss


def sample_triplets(g, n, etype=('user', 'interacts', 'item'), device='cpu'):
    # 1. All positive samples.
    users, pos_items = g.edges(etype=etype)
    num_edges = users.shape[0]
    num_items = g.num_nodes('item')

    # 2. Randomly sample n positive edges (if the number of edges is less than n, sample all)
    n = min(n, num_edges)
    perm = torch.randperm(num_edges, device=users.device)[:n]
    u = users[perm]
    pos_i = pos_items[perm]
 
    # 3.  For convenience in negative sampling, first construct an adjacency list
    user_neighbor_items = [[] for _ in range(g.num_nodes('user'))]
    all_u, all_i = users.tolist(), pos_items.tolist()
    for uu, ii in zip(all_u, all_i):
        user_neighbor_items[uu].append(ii)
    user_neighbor_sets = [set(neis) for neis in user_neighbor_items]

    # 4. for every tuple (user, pos_item), find one negative_item
    neg_i_list = []
    for uu in u.tolist():
        while True:
            cand = torch.randint(0, num_items, (1,)).item()
            if cand not in user_neighbor_sets[uu]:
                neg_i_list.append(cand)
                break

    neg_i = torch.tensor(neg_i_list, device=u.device)

    # Return triplets (user, pos_item, neg_item)
    return u, pos_i, neg_i


def train_lightgcn(g, hetero_g,
                   emb_size, n_layers, batch_size,
                   lr, epochs, device):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    g = g.to(device)
    
    if hetero_g is not None:
        hetero_g = hetero_g.to(device)
        n_users = hetero_g.num_nodes('user')
        n_items = hetero_g.num_nodes('item')
    else:
        raise ValueError("Please provide hetero_g for sampling")
    
    model = LightGCN(n_users, n_items, emb_size, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = BPRLoss(model)
    
    # Negative Sampling.
    users, pos_i, neg_i = sample_triplets(hetero_g, hetero_g.num_edges(etype='interacts'))
    users = users.to(device)
    pos_i = pos_i.to(device)
    neg_i = neg_i.to(device)
    dataset = TensorDataset(users, pos_i, neg_i)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_samples = 0

        # Epoch start
        epoch_start = time.time()

        for batch_idx, (batch_user, batch_pos, batch_neg) in enumerate(loader):
            batch_start = time.time()  # Batch start

            batch_user = batch_user.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            all_emb = model(g)
            user_emb = all_emb[batch_user]
            pos_emb = all_emb[batch_pos + n_users]
            neg_emb = all_emb[batch_neg + n_users]

            loss, reg_loss = loss_fn(user_emb, pos_emb, neg_emb, batch_user, batch_pos, batch_neg)
            reg_loss = reg_loss*decay
            loss=loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item() * len(batch_user)
            num_samples += len(batch_user)

            # Batch end
            batch_time = time.time() - batch_start
            # print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(loader)}, Batch time: {batch_time:.4f}s, Loss: {loss.item():.4f}")

        # Epoch end
        epoch_time = time.time() - epoch_start
        total_loss /= num_samples if num_samples > 0 else 1
        print(f"Epoch {epoch+1}/{epochs} finished. Epoch time: {epoch_time:.2f}s, Average BPR Loss: {total_loss:.4f}")



parser = argparse.ArgumentParser(description="Go lightGCN")
parser.add_argument('--batch', type=int,default=4096,
                    help="the batch size for bpr loss training procedure")
parser.add_argument('--recdim', type=int,default=16,
                    help="the embedding size of lightGCN")
parser.add_argument('--layer', type=int,default=3,
                    help="the layer num of lightGCN")
parser.add_argument('--lr', type=float,default=0.001,
                    help="the learning rate")
parser.add_argument('--decay', type=float,default=1e-4,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--testbatch', type=int,default=100,
                    help="the batch size of users for testing")
parser.add_argument('--dataset', type=str,default='gowalla',
                    help="available datasets: [gowalla, yelp2018, amazon-book]")
parser.add_argument('--path', type=str,default="./checkpoints",
                    help="path to save weights")
parser.add_argument('--topks', nargs='?',default="[20]",
                    help="@k test list")
parser.add_argument('--load', type=int,default=0)
parser.add_argument('--epochs', type=int,default=10)
parser.add_argument('--device', type=str,default='cpu')
parser.add_argument('--seed', type=int, default=2020, help='random seed')


if __name__ == "__main__":

    args = parser.parse_args()

    Latent_dim = args.recdim
    n_layers = args.layer
    lr=args.lr
    decay=args.decay
    batch_size=args.batch
    path = args.dataset
    epochs = args.epochs
    device = args.device
    # load dataset
    data = DGLDataset(path)
    train_lightgcn(data.graph, data.hetero_graph, Latent_dim, n_layers, batch_size, lr, epochs, device)