import wandb
import numpy as np
import scipy.sparse as sp
import math
from SUBCDR.model.crossdomain_recommender import CrossDomainRecommender
import os
from SUBCDR.model.cross_domain_recommender.han_conv import HANConv
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Subcdr(CrossDomainRecommender):

    def __init__(self, config, dataset):
        super(Subcdr, self).__init__(config, dataset)

        #params = args.params
        # similar experiments's params
        self.out_dim = config["out_dim"]
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        #idea
        self.global_lambda_gate = config["global_lambda_gate"]
        self.graph_layer = config['graph_layer']

        self.device = config["device"]
        self.dtype = torch.float64
        #self.wandb = config["wandb
        self.neg_valid_num = config["neg_valid_num"]
        self.emb_dim = 128
        self.shared_dim = 64
        self.n_layers = 2
        self.reg_weight = 0.001
        self.drop_rate = 0.3
        self.neg_ratio = 1

        self.n_shared_users = self.overlapped_num_users
        self.d1_n_users, self.d2_n_users = self.source_num_users, self.target_num_users
        self.d1_n_items, self.d2_n_items = self.source_num_items, self.target_num_items
        self.n_users = self.total_num_users
        self.n_items = self.total_num_items

        self.dropout = nn.Dropout(self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

        self.d1_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)

        self.d1_item = nn.Embedding(self.n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item = nn.Embedding(self.n_items, self.emb_dim, dtype=self.dtype)


        self.d1_adj, self.d1_user_degree = dataset.inter_matrix(form='coo', value_field=None, domain='source')
        self.d2_adj, self.d2_user_degree = dataset.inter_matrix(form='coo', value_field=None, domain='target')
        self.d1_adj = self.d1_adj.astype(np.float64)
        self.d1_user_degree = torch.tensor(self.d1_user_degree, dtype=self.dtype)
        self.d2_adj = self.d2_adj.astype(np.float64)
        self.d2_user_degree = torch.tensor(self.d2_user_degree, dtype=self.dtype)
        self.d1_norm_adj = self.get_norm_adj_mat_t(self.d1_adj, mode="src").to(self.device)
        self.d2_norm_adj = self.get_norm_adj_mat_t(self.d2_adj, mode="tgt").to(self.device)

        if self.graph_layer in ['han-v3']:
            #user - item
            self.user_item_shared_graph = dataset.merge_adj_matrices(self.d1_adj, self.d2_adj)
            self.user_item_shared_graph = self.user_item_shared_graph.astype(np.float64)
            self.user_item_shared_graph = self.get_norm_adj_mat_t(self.user_item_shared_graph, mode="both").to(
                self.device)
            self.d1_norm_adj = self.user_item_shared_graph
            self.d2_norm_adj = self.user_item_shared_graph
            # user-user
            value, row_indices, col_indices, aspects = dataset.process_user_file(self.user_user_file, dataset.source_user_ID_remap_dict, dataset.target_user_ID_remap_dict)
            self.aspect_types = list(set(aspects))

            if self.graph_layer == 'han-v3':
                edge_types = [('user_tgt', f'similar_{e}', 'user_src') for e in self.aspect_types] + [('user_src', f'similar_{e}', 'user_tgt') for e in self.aspect_types]

                self.metadata = (
                    ['user_src', 'user_tgt', 'item_src', 'item_tgt'],  
                    edge_types
                    )
                
                self.edge_index_dict = {}
                for e in self.aspect_types:
                    mask = aspects==e
                    sub_graph = coo_matrix((value[mask], (row_indices[mask], col_indices[mask])), shape=(self.n_users, self.n_users))
                    norm_graph = self.get_norm_adj_mat_t(sub_graph, mode='han').coalesce().indices().to(self.device)
                    self.edge_index_dict[('user_src', f'similar_{e}', 'user_tgt')] = norm_graph
                    self.edge_index_dict[('user_tgt', f'similar_{e}', 'user_src')] = torch.flip(norm_graph, [0])

            self.han_conv = HANConv(
                in_channels={'user_src': self.emb_dim // 2, 'user_tgt': self.emb_dim// 2, 'item_src': self.emb_dim// 2, 'item_tgt': self.emb_dim// 2}, 
                out_channels=self.emb_dim// 2,
                metadata=self.metadata,
                heads=8,
                negative_slope=0.2,
                dropout=0.2
            )


        user_laplace = self.d1_user_degree + self.d2_user_degree + 1e-7

        self.d1_user_degree = (self.d1_user_degree / user_laplace).to(dtype=self.dtype, device=self.device).unsqueeze(1)
        self.d2_user_degree = (self.d2_user_degree / user_laplace).to(dtype=self.dtype, device=self.device).unsqueeze(1)
        self.apply(self.xavier_init)

    def get_norm_adj_mat_t(self, interaction_matrix, mode="src"):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} /times A /times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        A = sp.dok_matrix(
                (self.total_num_users + self.total_num_items, self.total_num_users + self.total_num_items),
                dtype=np.float64
            )
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        if mode == "both" or mode == "han":
            data_dict = dict(
                zip(zip(inter_M.row, inter_M.col), [1] * inter_M.nnz)
            )
            data_dict.update(
                dict(
                    zip(
                        zip(inter_M_t.row, inter_M_t.col),
                        [1] * inter_M_t.nnz,
                    )
                )
            )
        else:
            data_dict = dict(
                zip(zip(inter_M.row, inter_M.col + self.total_num_users), [1] * inter_M.nnz)
            )
            data_dict.update(
                dict(
                    zip(
                        zip(inter_M_t.row + self.total_num_users, inter_M_t.col),
                        [1] * inter_M_t.nnz,
                    )
                )
            )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        SparseL = SparseL.to(dtype=self.dtype)
        return SparseL

    def xavier_init(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def get_emb(self):
        d1_user = self.d1_user.weight
        d1_item = self.d1_item.weight
        d1_emb = torch.cat([d1_user, d1_item])

        d2_user = self.d2_user.weight
        d2_item = self.d2_item.weight
        d2_emb = torch.cat([d2_user, d2_item])

        return d1_emb, d2_emb

    def graph_gcn_layer(self, adj_matrix, all_emb):
        side_emb = torch.sparse.mm(adj_matrix, all_emb)
        new_emb = side_emb + torch.mul(all_emb, side_emb)
        new_emb = all_emb + new_emb
        new_emb = self.dropout(new_emb)
        return new_emb

  
    def transfer_han_v3_layer(self, d1_emb, d2_emb):
        d1_user_emb, d1_item_emb = torch.split(d1_emb, [self.n_users, self.n_items])
        d2_user_emb, d2_item_emb = torch.split(d2_emb, [self.n_users, self.n_items])


        x_dict = {
            'user_src': d1_user_emb,
            'user_tgt': d2_user_emb
        }

        out_dict, weight = self.han_conv(x_dict, self.edge_index_dict, return_semantic_attention_weights=True)

        d1_user_emb_common, d2_user_emb_common = out_dict['user_src'], out_dict['user_tgt']

        d1_user_emb = (d1_user_emb_common + d1_user_emb) / 2
        d2_user_emb = (d2_user_emb_common + d2_user_emb) / 2

        d1_all_emb = torch.cat([d1_user_emb, d1_item_emb], dim=0)
        d2_all_emb = torch.cat([d2_user_emb, d2_item_emb], dim=0)

        return d1_all_emb, d2_all_emb



    def graph_transfer(self, d1_emb, d2_emb):
        d1_sh, d1_sp = torch.split(d1_emb, [self.shared_dim, self.emb_dim - self.shared_dim], -1)
        d2_sh, d2_sp = torch.split(d2_emb, [self.shared_dim, self.emb_dim - self.shared_dim], -1)

        d1_graph_sh = self.graph_gcn_layer(self.d1_norm_adj, d1_sh)
        d1_graph_sp = self.graph_gcn_layer(self.d1_norm_adj, d1_sp)
        d2_graph_sh = self.graph_gcn_layer(self.d2_norm_adj, d2_sh)
        d2_graph_sp = self.graph_gcn_layer(self.d2_norm_adj, d2_sp)
        d1_trans_sh, d2_trans_sh = self.transfer_han_v3_layer(d1_graph_sh, d2_graph_sh)
        d1_trans_sp, d2_trans_sp = self.transfer_han_v3_layer(d1_graph_sp, d2_graph_sp)

        d1_norm_sh = F.normalize(d1_trans_sh, 2, -1)
        d1_norm_sp = F.normalize(d1_trans_sp, 2, -1)
        d2_norm_sh = F.normalize(d2_trans_sh, 2, -1)
        d2_norm_sp = F.normalize(d2_trans_sp, 2, -1)
        d1_norm_emb = torch.cat([d1_norm_sh, d1_norm_sp], -1)
        d2_norm_emb = torch.cat([d2_norm_sh, d2_norm_sp], -1)

        return d1_norm_emb, d2_norm_emb

    def extract_ui(self, emb_list, domain):
        emb = torch.cat(emb_list, -1)
        if domain == 0:
            d2_user = emb[:self.d2_n_users]
            d2_item = emb[self.n_users:self.n_users + self.d2_n_items]
            d1_user = torch.cat([emb[:self.n_shared_users], emb[self.d2_n_users:self.n_users]])
            d1_item = torch.cat([emb[self.n_users:self.n_users+self.overlapped_num_items], emb[self.n_users+self.d2_n_items:]])
            return d1_user, d1_item, d2_user, d2_item
        elif domain == 2:
            final_user = emb[:self.d2_n_users]
            final_item = emb[self.n_users:self.n_users + self.d2_n_items]
        else:
            final_user = torch.cat([emb[:self.n_shared_users], emb[self.d2_n_users:self.n_users]])
            final_item = torch.cat([emb[self.n_users:self.n_users+self.overlapped_num_items], emb[self.n_users+self.d2_n_items:]])
        return final_user, final_item

    def forward(self):
        d1_emb, d2_emb = self.get_emb()

        d1_emb_list = [d1_emb]
        d2_emb_list = [d2_emb]

        # Graph transfer
        for i in range(self.n_layers):
            d1_emb, d2_emb = self.graph_transfer(d1_emb, d2_emb)
            d1_emb_list.append(d1_emb)
            d2_emb_list.append(d2_emb)
        d1_user, d1_item = self.extract_ui(d1_emb_list, 1)
        d2_user, d2_item = self.extract_ui(d2_emb_list, 2)

        return d1_user, d1_item, d2_user, d2_item

    def get_score(self, user_emb, item_emb):
        scores = torch.mul(user_emb, item_emb).sum(dim=-1)
        return scores

    def get_reg_loss(self, user_ids, item_ids, domain):
        d1_emb, d2_emb = self.get_emb()
        if domain == 1:
            user_emb = d1_emb[:self.n_users]
            item_emb = d1_emb[self.n_users:]
        else:
            user_emb = d2_emb[:self.n_users]
            item_emb = d2_emb[self.n_users:]

        user_reg_loss = (torch.norm(user_emb[user_ids]) ** 2 + torch.norm(user_emb[user_ids]) ** 2) / len(user_ids)
        item_reg_loss = (torch.norm(item_emb[item_ids]) ** 2 + torch.norm(item_emb[item_ids]) ** 2) / len(item_ids)

        reg_loss = user_reg_loss + item_reg_loss

        return reg_loss

    def get_split(self, emb):
        sh, sp = torch.split(emb, [self.shared_dim * (self.n_layers + 1),
                                   (self.emb_dim - self.shared_dim) * (self.n_layers + 1)], -1)
        return sh, sp
    def calculate_loss_subcdr(self, interaction):
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb = self.forward()

        d1_user_emb_all = torch.cat([
            d1_user_emb[:self.n_shared_users],
            d2_user_emb[self.n_shared_users:],
            d1_user_emb[self.n_shared_users:]
        ], dim=0)
        d2_user_emb_all = torch.cat([
            d2_user_emb,
            d1_user_emb[self.n_shared_users:]
        ], dim=0)
        d_item_emb = torch.cat([
            d2_item_emb,
            d1_item_emb[self.overlapped_num_items:]
        ], dim=0)

        d1_users, d1_items = d1_user_emb_all[interaction[self.SOURCE_USER_ID]], d_item_emb[interaction[self.SOURCE_ITEM_ID]]
        d1_label = interaction[self.SOURCE_LABEL].to(dtype=self.dtype, device=self.device)
        d1_output = self.sigmoid(self.get_score(d1_users, d1_items))
        d1_loss = self.loss(d1_output, d1_label)
        d1_loss = d1_loss + self.reg_weight * self.get_reg_loss(interaction[self.SOURCE_USER_ID], interaction[self.SOURCE_ITEM_ID], 1)

        # Domain 2 Positive inters
        d2_users, d2_items = d2_user_emb_all[interaction[self.TARGET_USER_ID]], d_item_emb[interaction[self.TARGET_ITEM_ID]]
        d2_label = interaction[self.TARGET_LABEL].to(dtype=self.dtype, device=self.device)
        d2_output = self.sigmoid(self.get_score(d2_users, d2_items))
        d2_loss = self.loss(d2_output, d2_label)
        d2_loss = d2_loss + self.reg_weight * self.get_reg_loss(interaction[self.TARGET_USER_ID], interaction[self.TARGET_ITEM_ID], 2)

        d1_neg_label = torch.zeros_like(interaction[self.SOURCE_LABEL], dtype=self.dtype, device=self.device)
        d2_neg_label = torch.zeros_like(interaction[self.TARGET_LABEL], dtype=self.dtype, device=self.device)
        for i in range(self.neg_ratio):
            # Negative samples
            # d1_neg_items = d1_item_emb[neg_d1_item[:, i]]
            d1_neg_items = d_item_emb[interaction[self.SOURCE_NEG_ITEM_ID]]
            d1_neg_output = self.sigmoid(self.get_score(d1_users, d1_neg_items))
            d1_loss = d1_loss + self.loss(d1_neg_output, d1_neg_label)
            d1_loss = d1_loss + self.reg_weight * self.get_reg_loss(interaction[self.SOURCE_USER_ID], interaction[self.SOURCE_NEG_ITEM_ID], 1)

            # d2_neg_items = d2_item_emb[neg_d2_item[:, i]]
            d2_neg_items = d_item_emb[interaction[self.TARGET_NEG_ITEM_ID]]
            d2_neg_output = self.sigmoid(self.get_score(d2_users, d2_neg_items))
            d2_loss = d2_loss + self.loss(d2_neg_output, d2_neg_label)
            d2_loss = d2_loss + self.reg_weight * self.get_reg_loss(interaction[self.TARGET_USER_ID], interaction[self.TARGET_NEG_ITEM_ID], 2)

        total_loss = (d1_loss + d2_loss) / (1 + self.neg_ratio)
            
        return total_loss

    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, mode):
        len_1 = len(eval_set_1.dataset)
        len_2 = len(eval_set_2.dataset)
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb = self.forward()
        hit_1, hit_10, hit_20, hit_50, ndcg_1, ndcg_10, ndcg_20, ndcg_50 = 0, 0, 0, 0, 0, 0, 0, 0
        for test_batch in eval_set_1:
            d1_users = test_batch[0][:, :1].repeat(1, 1 + self.neg_valid_num)
            d1_items = test_batch[0][:, 1:]
            scores = self.get_score(d1_user_emb[d1_users], d1_item_emb[d1_items])
            ranks = torch.sum(scores > scores[:, :1], dim=-1) + 1
            for i in range(len(ranks)):
                rank = ranks[i].item()
                if rank == 1:
                    hit_1 += 1
                    ndcg_1 += 1 / math.log2(rank + 1)
                if rank <= 10:
                    hit_10 += 1
                    ndcg_10 += 1 / math.log2(rank + 1)
                if rank <= 20:
                    hit_20 += 1
                    ndcg_20 += 1 / math.log2(rank + 1)
                if rank <= 50:
                    hit_50 += 1
                    ndcg_50 += 1 / math.log2(rank + 1)

        hits_1, hits_10, hits_20, hits_50, ndcgs_1, ndcgs_10, ndcgs_20, ndcgs_50 = hit_1 / len_1, hit_10 / len_1, hit_20 / len_1, hit_50 / len_1, ndcg_1 / len_1, ndcg_10 / len_1, ndcg_20 / len_1, ndcg_50 / len_1

        print("Domain 1")
        print(
            f"Hit@1: {hits_1:.4f}, NDCG@1: {ndcgs_1:.4f},Hit@10: {hits_10:.4f}, NDCG@10: {ndcgs_10:.4f},Hit@20: {hits_20:.4f}, NDCG@20: {ndcgs_20:.4f}, Hit@50: {hits_50:.4f}, NDCG@50: {ndcgs_50:.4f}")

        if self.wandb:
            wandb.log({f"D1-data5": {
                "Hit@1": hits_1, "NDCG@1": ndcgs_1,
                "Hit@10": hits_10, "NDCG@10": ndcgs_10,
                "Hit@20": hits_20, "NDCG@20": ndcgs_20,
                "Hit@50": hits_50, "NDCG@50": ndcgs_50
            }})

        hit_1, hit_10, hit_20, hit_50, ndcg_1, ndcg_10, ndcg_20, ndcg_50 = 0, 0, 0, 0, 0, 0, 0, 0
        for test_batch in eval_set_2:
            d2_users = test_batch[0][:, :1].repeat(1, 1 + self.neg_valid_num)
            d2_items = test_batch[0][:, 1:]
            scores = self.get_score(d2_user_emb[d2_users], d2_item_emb[d2_items])
            ranks = torch.sum(scores > scores[:, :1], dim=-1) + 1
            for i in range(len(ranks)):
                rank = ranks[i].item()
                if rank == 1:
                    hit_1 += 1
                    ndcg_1 += 1 / math.log2(rank + 1)
                if rank <= 10:
                    hit_10 += 1
                    ndcg_10 += 1 / math.log2(rank + 1)
                if rank <= 20:
                    hit_20 += 1
                    ndcg_20 += 1 / math.log2(rank + 1)
                if rank <= 50:
                    hit_50 += 1
                    ndcg_50 += 1 / math.log2(rank + 1)

        hits_1, hits_10, hits_20, hits_50, ndcgs_1, ndcgs_10, ndcgs_20, ndcgs_50 = hit_1 / len_2, hit_10 / len_2, hit_20 / len_2, hit_50 / len_2, ndcg_1 / len_2, ndcg_10 / len_2, ndcg_20 / len_2, ndcg_50 / len_2

        print("Domain 2")
        print(
            f"Hit@1: {hits_1:.4f}, NDCG@1: {ndcgs_1:.4f},Hit@10: {hits_10:.4f}, NDCG@10: {ndcgs_10:.4f},Hit@20: {hits_20:.4f}, NDCG@20: {ndcgs_20:.4f}, Hit@50: {hits_50:.4f}, NDCG@50: {ndcgs_50:.4f}")

        if self.wandb:
            wandb.log({f"D2-data5": {
                "Hit@1": hits_1, "NDCG@1": ndcgs_1,
                "Hit@10": hits_10, "NDCG@10": ndcgs_10,
                "Hit@20": hits_20, "NDCG@20": ndcgs_20,
                "Hit@50": hits_50, "NDCG@50": ndcgs_50
            }})
    def neg_sort_predict(self, inter, test_neg=None, valid_neg=None):
        d1_user_emb, d1_item_emb, d2_user_emb, d2_item_emb = self.forward()

        d_item_emb = torch.cat([d2_item_emb, d1_item_emb[self.overlapped_num_items:]], dim=0)

        user_ids = inter[self.TARGET_USER_ID] if self.TARGET_USER_ID in inter else inter[self.SOURCE_USER_ID]
        domain = "tgt" if self.TARGET_USER_ID in inter else "src"

        user_src = d1_user_emb[user_ids]
        user_tgt = d2_user_emb[user_ids]


        batch_pos_test = torch.tensor([test_neg[u.item()]['positive'] for u in user_ids]).to(self.device)
        batch_neg_test = torch.tensor([test_neg[u.item()]['negative'] for u in user_ids]).to(self.device)

        batch_pos_valid = torch.tensor([valid_neg[u.item()]['positive'] for u in user_ids]).to(self.device)
        batch_neg_valid = torch.tensor([valid_neg[u.item()]['negative'] for u in user_ids]).to(self.device)


        batch_pos_test_emb = d_item_emb[batch_pos_test]
        batch_neg_test_emb = d_item_emb[batch_neg_test.view(-1)].view(len(user_ids), 99, -1)
        batch_pos_valid_emb = d_item_emb[batch_pos_valid]
        batch_neg_valid_emb = d_item_emb[batch_neg_valid.view(-1)].view(len(user_ids), 99, -1)


        test_item_emb = torch.cat([batch_pos_test_emb.unsqueeze(1), batch_neg_test_emb], dim=1)
        valid_item_emb = torch.cat([batch_pos_valid_emb.unsqueeze(1), batch_neg_valid_emb], dim=1)


        if domain == "src":
            test = self.sigmoid(torch.matmul(user_src.unsqueeze(1), test_item_emb.transpose(1, 2)).squeeze(1))
            valid = self.sigmoid(torch.matmul(user_src.unsqueeze(1), valid_item_emb.transpose(1, 2)).squeeze(1))
        else:
            test = self.sigmoid(torch.matmul(user_tgt.unsqueeze(1), test_item_emb.transpose(1, 2)).squeeze(1))
            valid = self.sigmoid(torch.matmul(user_tgt.unsqueeze(1), valid_item_emb.transpose(1, 2)).squeeze(1))

        return test, valid
