import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import math
import wandb
import pickle


class AbtCDR(nn.Module):

    def __init__(self, args):
        super(AbtCDR, self).__init__()

        self.device = args.device
        self.latent_dim = 64
        self.n_layers = 3
        self.reg_weight = 0.00001
        self.drop_rate = 0.1
        self.connect_way = 'concat'
        self.t = 5

        self.source_num_users = args.d1['n_users']
        self.target_num_users = args.d2['n_users']
        self.source_num_items = args.d1['n_items']
        self.target_num_items = args.d2['n_items']
        self.n_fold = 1
        self.wandb = args.wandb

        #GAI
        self.neg_valid_num = args.neg_valid_num
        self.dtype = torch.float32
        self.n_shared_users = args.data['n_shared_users']
        self.n_users = args.d1['n_users'] + args.d2['n_users'] - args.data['n_shared_users']
        self.n_items = args.d1['n_items'] + args.d2['n_items']

        self.n_interaction = 5

        self.source_user_embedding = torch.nn.Parameter(torch.empty(self.n_users, self.latent_dim, dtype=self.dtype))
        self.target_user_embedding = torch.nn.Parameter(torch.empty(self.n_users, self.latent_dim, dtype=self.dtype))

        self.source_item_embedding = torch.nn.Parameter(torch.empty(self.source_num_items, self.latent_dim, dtype=self.dtype))
        self.target_item_embedding = torch.nn.Parameter(torch.empty(self.target_num_items, self.latent_dim, dtype=self.dtype))

        self.mapping = torch.nn.Parameter(torch.empty(self.latent_dim, self.latent_dim, dtype=self.dtype))

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()

        self.norm_adj_s = args.d1["inter_mat"]
        self.norm_adj_t = args.d2["inter_mat"]
        self.norm_adj_s, self.d1_user_degree = self.get_norm_adj(self.norm_adj_s)
        self.norm_adj_t, self.d2_user_degree = self.get_norm_adj(self.norm_adj_t)
        combined_degrees = torch.empty((self.n_users, 2), device=self.device)
        user_laplace = self.d1_user_degree + self.d2_user_degree + 1e-7
        self.d1_user_degree = (self.d1_user_degree / user_laplace).to(dtype=self.dtype).unsqueeze(1)
        self.d2_user_degree = (self.d2_user_degree / user_laplace).to(dtype=self.dtype).unsqueeze(1)
        combined_degrees[:, 0] = self.d1_user_degree
        combined_degrees[:, 1] = self.d2_user_degree
        self.domain_laplace = combined_degrees

        self.target_restore_user_e = None
        self.target_restore_item_e = None

        torch.nn.init.xavier_normal_(self.source_user_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.target_user_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.source_item_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.target_item_embedding, gain=1)
        torch.nn.init.xavier_normal_(self.mapping, gain=1)

        self.all_weights = torch.nn.ModuleList()

        self.all_weights.append(torch.nn.Linear(64, 128))
        self.all_weights.append(torch.nn.Linear(128, 128))
        self.all_weights.append(torch.nn.Linear(128, 64))
    def get_norm_adj(self, adj_mat):
        rowsum = np.array(adj_mat.sum(1)).flatten()
        users_degree = torch.from_numpy(rowsum[:self.n_users]).to(self.device)
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        adj = adj_mat.dot(r_mat_inv).transpose().dot(r_mat_inv)
        adj = adj.tocoo()
        indices = torch.LongTensor(np.array([adj.row, adj.col]))
        data = torch.tensor(adj.data, dtype=self.dtype)
        norm_adj = torch.sparse_coo_tensor(indices, data, torch.Size(adj.shape), dtype=self.dtype, device=self.device)

        return norm_adj, users_degree
    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding
            item_embeddings = self.source_item_embedding
            norm_adj_matrix = self.norm_adj_s
        else:
            user_embeddings = self.target_user_embedding
            item_embeddings = self.target_item_embedding
            norm_adj_matrix = self.norm_adj_t
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def _split_A_hat(self, X, n_items):
        A_fold_hat = []
        fold_len = (self.source_num_users + n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.source_num_users + n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(scipy_sparse_mat_to_torch_sparse_tensor(X[start:end]))
        return A_fold_hat

    def graph_layer(self, adj_matrix, all_embeddings):
        side_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        return side_embeddings

    def inter_embedding(self, source_all_embeddings, target_all_embeddings):

        source_user_embeddings, source_item_embeddings = torch.split(source_all_embeddings,
                                                                     [self.n_users, self.source_num_items])
        target_user_embeddings, target_item_embeddings = torch.split(target_all_embeddings,
                                                                     [self.n_users, self.target_num_items])
        source_user_full = source_user_embeddings.clone()
        target_user_full = target_user_embeddings.clone()
        src_user = source_user_embeddings[:self.n_shared_users]  # [n, embed_dim]
        tgt_user = target_user_embeddings[:self.n_shared_users]  # [n, embed_dim]
        src_user_raw = src_user.clone()
        tgt_user_raw = tgt_user.clone()
        a = torch.matmul(src_user, self.mapping)

        s = torch.exp(torch.matmul(a, tgt_user.T) / self.t)

        sr = F.normalize(s, p=1, dim=1)
        sc = F.normalize(s, p=1, dim=0)

        src_user = (src_user + torch.matmul(sr, tgt_user))  # [n, embed_dim]
        tgt_user = (tgt_user + torch.matmul(sc.T, src_user))  # [n, embed_dim]

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single_(adj):
            degree = torch.sum(adj, dim=1)
            degree_matrix = torch.diag(degree)
            norm_adj = torch.inverse(degree_matrix) @ adj
            return norm_adj

        adj_s = torch.matmul(s, s.T)  # [n, n]
        adj_s = normalized_adj_single_(adj_s + torch.eye(self.n_shared_users).cuda())  
        adj_t = torch.matmul(s.T, s)  # [n, n]
        adj_t = normalized_adj_single_(adj_t + torch.eye(self.n_shared_users).cuda())

        for _ in range(3):
            src_user = torch.mm(adj_s, src_user)  # [n, embed_dim]
            src_user = torch.nn.ReLU()(src_user)
            src_user = F.normalize(src_user, p=2, dim=1)

            tgt_user = torch.mm(adj_t, tgt_user)  # [n, embed_dim]
            tgt_user = torch.nn.ReLU()(tgt_user)
            tgt_user = F.normalize(tgt_user, p=2, dim=1)

        src_user = (src_user_raw + src_user) / 2  # [n, embed_dim]
        tgt_user = (tgt_user_raw + tgt_user) / 2  # [n, embed_dim]

        source_user_full[:self.n_shared_users] = src_user  
        target_user_full[:self.n_shared_users] = tgt_user

        source_alltransfer_embeddings = torch.cat(
            [source_user_full, source_item_embeddings], 0)
        target_alltransfer_embeddings = torch.cat(
            [target_user_full, target_item_embeddings], 0)

        return source_alltransfer_embeddings, target_alltransfer_embeddings

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        for k in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)

            import random
            if random.random() < 0.3:
                source_all_embeddings, target_all_embeddings = self.inter_embedding(source_all_embeddings,
                                                                                    target_all_embeddings)

            source_norm_embeddings = nn.functional.normalize(source_all_embeddings, p=2, dim=1)
            target_norm_embeddings = nn.functional.normalize(target_all_embeddings, p=2, dim=1)
            source_embeddings_list.append(source_norm_embeddings)
            target_embeddings_list.append(target_norm_embeddings)

        if self.connect_way == 'concat':
            source_lightgcn_all_embeddings = torch.cat(source_embeddings_list, 1)
            target_lightgcn_all_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_lightgcn_all_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_lightgcn_all_embeddings = torch.mean(source_lightgcn_all_embeddings, dim=1)
            target_lightgcn_all_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_lightgcn_all_embeddings = torch.mean(target_lightgcn_all_embeddings, dim=1)

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                             [self.n_users,
                                                                              self.source_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                             [self.n_users,
                                                                              self.target_num_items])

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings

    def calculate_single_loss(self, user, item, label, flag):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        if flag == "source":
            source_u_embeddings = source_user_all_embeddings[user]
            source_i_embeddings = source_item_all_embeddings[item]
            source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
            source_bce_loss = self.loss(source_output, torch.from_numpy(label).cuda().to(torch.float))

            source_reg_loss = self.reg_loss(source_u_embeddings, source_i_embeddings)

            source_loss = source_bce_loss + self.reg_weight * source_reg_loss

            return source_loss, 0

        if flag == "target":
            target_u_embeddings = target_user_all_embeddings[user]
            target_i_embeddings = target_item_all_embeddings[item]

            target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
            target_bce_loss = self.loss(target_output, torch.from_numpy(label).cuda().to(torch.float))

            target_reg_loss = self.reg_loss(target_u_embeddings, target_i_embeddings)

            target_loss = target_bce_loss + self.reg_weight * target_reg_loss
            return 0, target_loss

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        source_u_embeddings = source_user_all_embeddings[inter_d1[:, 0]]
        source_i_embeddings = source_item_all_embeddings[inter_d1[:, 1]]
        target_u_embeddings = target_user_all_embeddings[inter_d2[:, 0]]
        target_i_embeddings = target_item_all_embeddings[inter_d2[:, 1]]

        # calculate BCE Loss in source domain
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        d1_label = inter_d1[:, 2].to(dtype=self.dtype, device=self.device)
        source_bce_loss = self.loss(source_output, d1_label)

        # calculate Reg Loss in source domain
        source_reg_loss = self.reg_loss(source_u_embeddings, source_i_embeddings)

        source_loss = source_bce_loss + self.reg_weight * source_reg_loss

        # calculate BCE Loss in target domain
        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        d2_label = inter_d2[:, 2].to(dtype=self.dtype, device=self.device)
        target_bce_loss = self.loss(target_output, d2_label)

        # calculate Reg Loss in target domain
        target_reg_loss = self.reg_loss(target_u_embeddings, target_i_embeddings)

        target_loss = target_bce_loss + self.reg_weight * target_reg_loss
        losses = source_loss + target_loss

        return losses
    def get_score(self, user_emb, item_emb):
        scores = torch.mul(user_emb, item_emb).sum(dim=-1)
        return scores
    def predict(self, eval_set_1, eval_set_2, mode):
        len_1 = len(eval_set_1.dataset)
        len_2 = len(eval_set_2.dataset)
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb = self.forward()

        hit_1, hit_5, hit_10, hit_20, hit_50, ndcg_1, ndcg_5, ndcg_10, ndcg_20, ndcg_50, mrr_1, mrr_5, mrr_10, mrr_20, mrr_50 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for test_batch in eval_set_1:
            d1_users = test_batch[0][:, :1].repeat(1, 1+self.neg_valid_num)
            d1_items = test_batch[0][:, 1:]
            scores = self.get_score(d1_user_emb[d1_users], d1_item_emb[d1_items])
            ranks = torch.sum(scores > scores[:, :1], dim=-1) + 1
            for i in range(len(ranks)):
                rank = ranks[i].item()
                if rank == 1:
                    hit_1 += 1
                    ndcg_1 += 1 / math.log2(rank+1)
                    mrr_1 += 1 / rank
                if rank <= 5:
                    hit_5 += 1
                    ndcg_5 += 1 / math.log2(rank+1)
                    mrr_5 += 1 / rank
                if rank <= 10:
                    hit_10 += 1
                    ndcg_10 += 1 / math.log2(rank+1)
                    mrr_10 += 1 / rank
                if rank <= 20:
                    hit_20 += 1
                    ndcg_20 += 1 / math.log2(rank+1)
                    mrr_20 += 1 / rank
                if rank <= 50:
                    hit_50 += 1
                    ndcg_50 += 1 / math.log2(rank+1)
                    mrr_50 += 1 / rank

        hits_1, hits_5, hits_10, hits_20, hits_50, ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_20, ndcgs_50, mrrs_1, mrrs_5, mrrs_10, mrrs_20, mrrs_50 = hit_1 / len_1, hit_5 / len_1, hit_10 / len_1, hit_20 / len_1, hit_50 / len_1, ndcg_1 / len_1, ndcg_5 / len_1, ndcg_10 / len_1, ndcg_20 / len_1, ndcg_50 / len_1, mrr_1 / len_1, mrr_5 / len_1, mrr_10 / len_1, mrr_20 / len_1, mrr_50 / len_1

        print("Domain 1")
        print(f"Hit@1: {hits_1:.4f}, NDCG@1: {ndcgs_1:.4f}, MRR@1: {mrrs_1:.4f}, Hit@5: {hits_5:.4f}, NDCG@5: {ndcgs_5:.4f}, MRR@5: {mrrs_5:.4f}, Hit@10: {hits_10:.4f}, NDCG@10: {ndcgs_10:.4f}, MRR@10: {mrrs_10:.4f}, Hit@20: {hits_20:.4f}, NDCG@20: {ndcgs_20:.4f}, MRR@20: {mrrs_20:.4f}, Hit@50: {hits_50:.4f}, NDCG@50: {ndcgs_50:.4f}, MRR@50: {mrrs_50:.4f}")

        if self.wandb:
            wandb.log({ f"D1-{mode}": {
                "Hit@1": hits_1, "NDCG@1": ndcgs_1, "MRR@1": mrrs_1,
                "Hit@5": hits_5, "NDCG@5": ndcgs_5, "MRR@5": mrrs_5,
                "Hit@10": hits_10, "NDCG@10": ndcgs_10, "MRR@10": mrrs_10,
                "Hit@20": hits_20, "NDCG@20": ndcgs_20, "MRR@20": mrrs_20,
                "Hit@50": hits_50, "NDCG@50": ndcgs_50, "MRR@50": mrrs_50
            }})

        hit_1, hit_5, hit_10, hit_20, hit_50, ndcg_1, ndcg_5, ndcg_10, ndcg_20, ndcg_50, mrr_1, mrr_5, ndcg_10, mrr_20, mrr_50 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for test_batch in eval_set_2:
            d2_users = test_batch[0][:, :1].repeat(1, 1+self.neg_valid_num)
            d2_items = test_batch[0][:, 1:]
            scores = self.get_score(d2_user_emb[d2_users], d2_item_emb[d2_items])
            ranks = torch.sum(scores > scores[:, :1], dim=-1) + 1
            for i in range(len(ranks)):
                rank = ranks[i].item()
                if rank == 1:
                    hit_1 += 1
                    ndcg_1 += 1 / math.log2(rank+1)
                    mrr_1 += 1 / rank
                if rank <= 5:
                    hit_5 += 1
                    ndcg_5 += 1 / math.log2(rank+1)
                    mrr_5 += 1 / rank
                if rank <= 10:
                    hit_10 += 1
                    ndcg_10 += 1 / math.log2(rank+1)
                    mrr_10 += 1 / rank
                if rank <= 20:
                    hit_20 += 1
                    ndcg_20 += 1 / math.log2(rank+1)
                    mrr_20 += 1 / rank
                if rank <= 50:
                    hit_50 += 1
                    ndcg_50 += 1 / math.log2(rank+1)
                    mrr_50 += 1 / rank

        hits_1, hits_5, hits_10, hits_20, hits_50, ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_20, ndcgs_50, mrrs_1, mrrs_5, mrrs_10, mrrs_20, mrrs_50 = hit_1 / len_2, hit_5 / len_2, hit_10 / len_2, hit_20 / len_2, hit_50 / len_2, ndcg_1 / len_2, ndcg_5 / len_2, ndcg_10 / len_2, ndcg_20 / len_2, ndcg_50 / len_2, mrr_1 / len_2, mrr_5 / len_2, mrr_10 / len_2, mrr_20 / len_2, mrr_50 / len_2

        print("Domain 2")
        print(f"Hit@1: {hits_1:.4f}, NDCG@1: {ndcgs_1:.4f}, MRR@1: {mrrs_1:.4f}, Hit@5: {hits_5:.4f}, NDCG@5: {ndcgs_5:.4f}, MRR@5: {mrrs_5:.4f}, Hit@10: {hits_10:.4f}, NDCG@10: {ndcgs_10:.4f}, MRR@10: {mrrs_10:.4f}, Hit@20: {hits_20:.4f}, NDCG@20: {ndcgs_20:.4f}, MRR@20: {mrrs_20:.4f}, Hit@50: {hits_50:.4f}, NDCG@50: {ndcgs_50:.4f}, MRR@50: {mrrs_50:.4f}")

        if self.wandb:
            wandb.log({ f"D2-{mode}": {
                "Hit@1": hits_1, "NDCG@1": ndcgs_1, "MRR@1": mrrs_1,
                "Hit@5": hits_5, "NDCG@5": ndcgs_5, "MRR@5": mrrs_5,
                "Hit@10": hits_10, "NDCG@10": ndcgs_10, "MRR@10": mrrs_10,
                "Hit@20": hits_20, "NDCG@20": ndcgs_20, "MRR@20": mrrs_20,
                "Hit@50": hits_50, "NDCG@50": ndcgs_50, "MRR@50": mrrs_50
            }})

    def full_sort_predict(self, user):
        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e


class EmbLoss(nn.Module):
    """
        EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
