'''
Created on March 1st, 2023

@author: Junkang Wu (jkwu0909@gmail.com)
'''
from tarfile import POSIX_MAGIC
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import sys
sys.path.append("..")
from utils import losses
from scipy.special import lambertw
from scipy.sparse import find


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1, eps=0.1):
        super(GraphConv, self).__init__()
        self.eps = eps
        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
        mess_dropout=True, edge_dropout=True, perturb = False):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]
        #print(item_embed)
        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        #print(all_embed)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                    self.edge_dropout_rate) if edge_dropout \
                                                                            else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            
            if perturb:
                random_noise = torch.rand_like(agg_embed).cuda()
                agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * self.eps
            
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)

        # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        #print(embs)
        return embs[:self.n_users, :], embs[self.n_users:, :]
        


class lgn_tau_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(lgn_tau_frame, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        #self.input = get_input_from_adjacency_matrix(self.adj_mat)
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.logger = logger
        if self.mess_dropout:
            self.dropout = nn.Dropout(args_config.mess_dropout_rate)
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs

        self.temperature = args_config.temperature
        self.temperature_2 = args_config.temperature_2

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.tau_mode = args_config.tau_mode

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.loss_name = args_config.loss_fn

        self.generate_mode = args_config.generate_mode

        if args_config.loss_fn == "Adap_tau_Loss":
            print(self.loss_name)
            print("start to make tables")
            self.lambertw_table = torch.FloatTensor(lambertw(np.arange(-1, 1002, 1e-4))).to(self.device)
            self.loss_fn = losses.Adap_tau_Loss()
        elif args_config.loss_fn == "SSM_Loss":
            print(self.loss_name)
            self.loss_fn = losses.SSM_Loss(self._margin, self.temperature, self._negativa_weight, args_config.pos_mode)
        else:
            raise NotImplementedError("loss={} is not support".format(args_config.loss_fn))

        self.register_buffer("memory_tau", torch.full((self.n_users,), 1 / 0.10))
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        self.reg_loss = losses.L2Loss()
        self.lamda = 1e+1#########################################################超参数定义

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    
    # def get_input_from_adjacency_matrix(adj_mat):
    #     # 获取非零元素的索引
    #     nonzero_indices = find(adj_mat)
    #     # 将行索引和列索引分别保存为数组
    #     user_indices = nonzero_indices[1]
    #     item_indices = nonzero_indices[0]
    #     # 构造输入数组，其中每个元素是一个包含用户索引和物品索引的元组
    #     inputs = torch.tensor([user_indices, item_indices])
    #     return inputs

    def _update_tau_memory(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau = x

    def _loss_to_tau(self, x, x_all):
        if self.tau_mode == "weight_v0":
            t_0 = x_all
            tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
        elif self.tau_mode == "weight_ratio":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                base_laberw = torch.quantile(x, self.temperature)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        elif self.tau_mode == "weight_mean":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
            else:
                base_laberw = torch.mean(x)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        return tau
    
    def sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    
    def reg_data_handler(self, user_online, item_online, user, item):
        #print(user_embed.shape)
        #print(item_embed.shape)
        # u_online = user_embed
        # i_online = item_embed
        # u_online, i_online = self.gcn(self.user_embed,
        #                              self.item_embed,
        #                              edge_dropout=True,
        #                              mess_dropout=True,
        #                              perturb=True)
        with torch.no_grad():
            u_target, i_target = user_online.clone(), item_online.clone()
            # u_target, i_target = self.gcn(self.user_embed,
            #                               self.item_embed,
            #                               edge_dropout=True,
            #                               mess_dropout=True,
            #                               perturb=True)
            # # edge pruning
            #print(u_target.shape)
            x = self.sparse_dropout(self.sparse_norm_adj)
            all_embeddings = torch.cat([u_target, i_target], 0)
            
            all_embeddings = all_embeddings.view(-1, all_embeddings.size(0))  # Reshape to [n_users + n_items, emb_size]
            #print(x.shape)
            #print(all_embeddings.shape)
            all_embeddings = torch.transpose(all_embeddings, 0, 1)
            random_noise = torch.rand_like(all_embeddings).cuda()
            all_embeddings += torch.sign(all_embeddings) * F.normalize(random_noise, dim=-1) * 0.1
            all_embeddings = torch.sparse.mm(x, all_embeddings)  # Transpose and perform sparse matrix multiplication
            u_target = all_embeddings[:self.n_users, :]
            i_target = all_embeddings[self.n_users:, :]
            
#             #print(u_target.shape)
#             #print(i_target.shape)
            u_target = u_target[user, :]
            i_target = i_target[item, :]

        u_online = user_online[user, :]
        i_online = item_online[item, :]
        return u_online, u_target, i_online, i_target
    
    def manual_normalize(self, input_tensor, dim=1, eps=1e-12):
        norm = torch.sqrt(torch.sum(input_tensor ** 2, dim=dim, keepdim=True) + eps)
        return input_tensor / norm
    
    
    
    def calculate_cf_loss(self, u_online, user_target, i_online, item_target, alpha=1.0, beta=1.0, temperature=0.5):
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        
        #  Multi-label Triplet Embeddings for Image Annotation from User-Generated Tags
        
        user_online = u_online.view(-1, u_online.size(0))
        item_online = i_online.view(-1, i_online.size(0))
        user_online = torch.transpose(user_online, 0, 1)
        item_online = torch.transpose(item_online, 0, 1)
        # Calculate similarity scores between user_online and item_online
        user_item_similarity = torch.matmul(user_online, item_online.t())  # Shape: (batch_size, batch_size)

        # Calculate similarity scores between user_target and item_target
        user_target_item_target_similarity = torch.matmul(user_target, item_target.t())  # Shape: (batch_size, batch_size)

        # Calculate similarity scores between user_target and item_online
        user_target_item_online_similarity = torch.matmul(user_target, item_online.t())  # Shape: (batch_size, batch_size)

        # Calculate similarity scores between user_online and item_target
        user_online_item_target_similarity = torch.matmul(user_online, item_target.t())  # Shape: (batch_size, batch_size)

        # Compute self-collaborative-filtering loss
        loss = F.relu(user_item_similarity - user_target_item_target_similarity).sum() + \
               F.relu(user_item_similarity - user_target_item_online_similarity).sum() + \
               F.relu(user_item_similarity - user_online_item_target_similarity).sum()

        return loss / (user_online.size(0) ** 2)  # Normalize the loss by batch size squared
#         u_online, i_online = self.predictor(u_online), self.predictor(i_online)
#         user_online = u_online.view(-1, u_online.size(0))
#         item_online = i_online.view(-1, i_online.size(0))
#         user_online = torch.transpose(user_online, 0, 1)
#         item_online = torch.transpose(item_online, 0, 1)

#         user_online_norm = F.normalize(user_online, dim=-1)
#         user_target_norm = F.normalize(user_target, dim=-1)
#         item_online_norm = F.normalize(item_online, dim=-1)
#         item_target_norm = F.normalize(item_target, dim=-1)

#         # Compute the similarity matrices
#         user_similarity_matrix = torch.matmul(user_online_norm, user_target_norm.T) / temperature
#         item_similarity_matrix = torch.matmul(item_online_norm, item_target_norm.T) / temperature

#         # Labels for positive pairs
#         batch_size = user_online.size(0)
#         user_labels = torch.arange(batch_size).to(user_online.device)
#         item_labels = torch.arange(batch_size).to(item_online.device)

#         # Contrastive loss using InfoNCE with hard negatives
#         contrastive_loss_user = F.cross_entropy(user_similarity_matrix, user_labels)
#         contrastive_loss_item = F.cross_entropy(item_similarity_matrix, item_labels)

#         # Reconstruction losses
#         recon_loss_user = F.mse_loss(user_online, user_target)
#         recon_loss_item = F.mse_loss(item_online, item_target)

#         # Total loss
#         loss = alpha * (recon_loss_user + recon_loss_item) + beta * (contrastive_loss_user + contrastive_loss_item)

#         return loss
    

    
    def forward(self, batch=None, loss_per_user=None, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout,
                                              perturb=False)
       
        # neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        u_online, u_target, i_online, i_target = self.reg_data_handler(user_gcn_emb, item_gcn_emb, user, pos_item)
       
        if s == 0 and w_0 is not None:
            # self.logger.info("Start to adjust tau with respect to users")
            tau_user = self._loss_to_tau(loss_per_user, w_0)
            self._update_tau_memory(tau_user)
        if self.sampling_method == "no_sample":
            return self.NO_Sample_Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], user, u_online, u_target, i_online, i_target, w_0)
        else:
            neg_item = batch['neg_items']
            return self.Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], item_gcn_emb[neg_item], user, u_online, u_target, i_online, i_target, w_0)


    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=-2)
        elif self.pool == 'sum':
            return embeddings.sum(dim=-2)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def gcn_emb(self):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        return user_gcn_emb.detach(), item_gcn_emb.detach()

    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if self.generate_mode == "cosine":
            user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)

        elif self.generate_mode == "reweight":
            # reweight focus on items
            item_norm = torch.norm(item_gcn_emb, p=2, dim=-1)
            mean_norm = item_norm.mean()
            item_gcn_emb = item_gcn_emb / item_norm.unsqueeze(1)  * mean_norm * self.reweight.unsqueeze(1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        cl_loss_sum = 0.0
        shape = view1.shape
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        
        view_1 = torch.squeeze(view1, dim=0)
        view_2 = torch.squeeze(view2, dim=0)
        #print(view_1)
        #print(view_2)
        pos_score = (view_1 * view_2).sum(dim=-1)
        #print(pos_score.shape)
        pos_score = torch.exp(pos_score / temperature)
        
        ttl_score = torch.matmul(view_1, view_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        
        return torch.mean(cl_loss)

    def cal_cl_loss(self, idx):
        #idx_cpu0 = idx[0]
        #idx_gpu0 = idx_cpu0.cuda()  # 将 idx[0] 移动到 GPU
        #u_idx = torch.unique(idx_gpu0.type(torch.long))
        #idx_cpu1 = idx[1]
        #idx_gpu1 = idx_cpu1.cuda()
        #i_idx = torch.unique(idx_gpu1.type(torch.long))
        #i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        #u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        #i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 将列表中的每个元素转换为 Tensor，并指定类型
        tensor_idx_0 = torch.tensor(idx[0]).type(torch.long)
        tensor_idx_1 = torch.tensor(idx[1]).type(torch.long)

        # 检查当前的设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将 Tensor 移动到正确的设备上
        tensor_idx_0 = tensor_idx_0.to(device)
        tensor_idx_1 = tensor_idx_1.to(device)

        # 获取唯一值并在 GPU 上进行计算
        u_idx = torch.unique(tensor_idx_0)
        i_idx = torch.unique(tensor_idx_1)
        user_view_1, item_view_1 = self.gcn(self.user_embed,
                                            self.item_embed,
                                            edge_dropout=self.edge_dropout,
                                            mess_dropout=self.mess_dropout,
                                            perturb=True)
        user_view_2, item_view_2 = self.gcn(self.user_embed,
                                            self.item_embed,
                                            edge_dropout=self.edge_dropout,
                                            mess_dropout=self.mess_dropout,
                                            perturb=True)
        #print(user_view_1)
        #print(user_view_2)
        #print(self.memory_tau)
        #temperature = self.memory_tau.mean()
        #print(temperature)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temperature)
        #print(user_cl_loss)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temperature)
        return user_cl_loss + item_cl_loss

    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user, u_online, u_target, i_online, i_target, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb) # [B, F]
        neg_e = self.pooling(neg_gcn_emb) # [B, M, F]

        item_e = torch.cat([pos_e.unsqueeze(1), neg_e], dim=1) # [B, M+1, F]
        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            item_e = F.normalize(item_e, dim=-1)

        y_pred = torch.bmm(item_e, u_e.unsqueeze(-1)).squeeze(-1) # [B M+1]
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2
                       + torch.norm(neg_gcn_emb[:, :, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        nce_loss = self.lamda * self.cal_cl_loss([u_e, pos_e])#######################计算loss
        cf_loss = self.calculate_cf_loss(u_online, u_target, i_online, i_target)
        #print(nce_loss)
        if self.loss_name == "Adap_tau_Loss":
            mask_zeros = None
            tau = torch.index_select(self.memory_tau, 0, user).detach()
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            return loss.mean() + emb_loss + nce_loss + cf_loss, loss_, emb_loss, tau, nce_loss + cf_loss
        elif self.loss_name == "SSM_Loss":
            loss, loss_ = self.loss_fn(y_pred)
            return loss.mean() + emb_loss + nce_loss + cf_loss, loss_, emb_loss, y_pred, nce_loss + cf_loss
        else:
            raise NotImplementedError("loss={} is not support".format(self.loss_name))


    def NO_Sample_Uniform_loss(self, user_gcn_emb, pos_gcn_emb, user, u_online, u_target, i_online, i_target, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb) # [B, F]

        if self.u_norm:
            u_e = F.normalize(u_e, dim=-1)
        if self.i_norm:
            pos_e = F.normalize(pos_e, dim=-1)
        # contrust y_pred framework
        row_swap = torch.cat([torch.arange(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        col_before = torch.cat([torch.arange(batch_size).long(), torch.zeros(batch_size).long()]).to(self.device)
        col_after = torch.cat([torch.zeros(batch_size).long(), torch.arange(batch_size).long()]).to(self.device)
        y_pred = torch.mm(u_e, pos_e.t().contiguous())
        y_pred[row_swap, col_before] = y_pred[row_swap, col_after]
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, :]) ** 2
                       + torch.norm(pos_gcn_emb[:, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        #nce_loss = self.lamda * self.cal_cl_loss([u_e, pos_e])#######################计算loss
        cf_loss = self.lamda * self.calculate_cf_loss(u_online, u_target, i_online, i_target, self.temperature)
        #print("No_Sample")
        #print(nce_loss)
        if self.loss_name == "Adap_tau_Loss":
            mask_zeros = None
            tau = torch.index_select(self.memory_tau, 0, user).detach()
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            return loss.mean() + emb_loss + cf_loss, loss_, emb_loss, tau, cf_loss
        elif self.loss_name == "SSM_Loss":
            loss, loss_ = self.loss_fn(y_pred)
            return loss.mean() + emb_loss + cf_loss, loss_, emb_loss, y_pred, cf_loss
        else:
            raise NotImplementedError("loss={} is not support".format(self.loss_name))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask