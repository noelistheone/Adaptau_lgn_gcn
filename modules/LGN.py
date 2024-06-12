"""
for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            # agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            # neighbors = torch.sparse.mm(interact_mat, agg_embed)
            # agg_embed = self.gat_layer(agg_embed, interact_mat)
            agg_embed = self.gc_layers(agg_embed, interact_mat)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            if perturbed:
                random_noise = torch.rand_like(agg_embed).cuda()
                agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * eps
            embs.append(agg_embed)
            if hop==0:
                all_embeddings_cl = agg_embed
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        if perturbed:
            return embs[:self.n_users, :], embs[self.n_users:, :],all_embeddings_cl[:self.n_users, :], all_embeddings_cl[self.n_users:, :]
        return embs[:self.n_users, :], embs[self.n_users:, :]
"""
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

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

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
                mess_dropout=True, edge_dropout=True, perturbed=True, eps=0.03):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            if perturbed:
                random_noise = torch.rand_like(agg_embed).cuda()
                agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * eps
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]

class lgn_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(lgn_frame, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

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
       
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.tau_mode = args_config.tau_mode
        
        self.alpha = 1.5
        
        self.user_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.Tanh()
        )
        self.dropout_rate = 0.1
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.min_eps = 0.01
        self.max_eps = 0.05
        self.beta = 0.1
        self.alpha = 0.01
        self.eps = 0.03
        self.eps_grad = 0  # Placeholder for gradient of epsilon
       
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
       
        self.loss_name = args_config.loss_fn
    
        self.generate_mode = args_config.generate_mode
        self.InfoNCE = losses.InfoNCE()

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
        self.register_buffer("memory_tau_u", torch.full((self.n_users,), 1 / 0.10))
        self.register_buffer("memory_tau_i", torch.full((self.n_items,), 1 / 0.10))
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        
        self.lamda = 1
        

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
    
    def update_eps(self, meta_loss_grad):
         with torch.no_grad():
            self.eps -= self.beta * meta_loss_grad
            self.eps = self.eps.clamp_(self.min_eps, self.max_eps)
        # self.eps_grad = meta_loss_grad
        # self.eps = max(self.min_eps, min(self.max_eps, self.eps - self.beta * self.eps_grad))
        # return self.eps

    def _update_tau_memory(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau = x
            
    def _update_tau_memory_u(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau_u = x

    def _update_tau_memory_i(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau_i = x

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
    
    
    
    def augment_embeddings(self, embeddings, rate=0.2):
        noise = torch.randn_like(embeddings) * rate
        return embeddings + noise
    
    def contrastive_loss(self, embeddings_1, embeddings_2, temperature=0.5):
        embeddings_1 = F.normalize(embeddings_1, dim=-1)
        embeddings_2 = F.normalize(embeddings_2, dim=-1)

        batch_size = embeddings_1.shape[0]

        # Compute similarities
        similarity_matrix = torch.mm(embeddings_1, embeddings_2.t()) / temperature

        # Mask to ignore self-comparisons
        mask = torch.eye(batch_size, device=embeddings_1.device).bool()

        # Compute positive similarity
        positive_similarity = similarity_matrix[mask].view(batch_size, -1)

        # Compute negative similarity
        negative_similarity = similarity_matrix[~mask].view(batch_size, -1)

        # Contrastive loss
        loss = -torch.log(torch.exp(positive_similarity) / torch.exp(negative_similarity).sum(dim=-1, keepdim=True)).mean()

        return loss


    def forward(self, batch=None, loss_per_user=None, loss_per_user_u = None, loss_per_user_i=None, meta_loss_grad=None, epoch=0, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        if meta_loss_grad is not None:
            self.update_eps(meta_loss_grad)
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout, perturbed=True, eps=self.eps)
        
        
        # neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        if s == 0 and w_0 is not None:
            # self.logger.info("Start to adjust tau with respect to users")
            tau_user = self._loss_to_tau(loss_per_user, w_0)
            tau_user_u = self._loss_to_tau(loss_per_user_u, w_0)
            tau_user_i = self._loss_to_tau(loss_per_user_i, w_0)
            self._update_tau_memory(tau_user)
            self._update_tau_memory_u(tau_user_u)
            self._update_tau_memory_i(tau_user_i)
        if self.sampling_method == "no_sample":
            return self.NO_Sample_Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], user, pos_item, w_0)
        else:
            neg_item = batch['neg_items']
            return self.Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], item_gcn_emb[neg_item], user, w_0)
        
        
       

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

    def gcn_emb(self, noise=False):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False
                                              )
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
    
    

    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user, w_0=None):
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

        if self.loss_name == "Adap_tau_Loss":
            mask_zeros = None
            tau = torch.index_select(self.memory_tau, 0, user).detach()
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            return loss.mean() + emb_loss, loss_, emb_loss, tau
        elif self.loss_name == "SSM_Loss":
            loss, loss_ = self.loss_fn(y_pred)
            return loss.mean() + emb_loss, loss_, emb_loss, y_pred
        else:
            raise NotImplementedError("loss={} is not support".format(self.loss_name))
            
    
            
#     def cal_cl_loss(self, user_gcn_emb, item_gcn_emb, user_idx, item_idx):
        
#         user_view_1, item_view_1 = self.gcn(self.user_embed,
#                                               self.item_embed,
#                                               edge_dropout=False,
#                                               mess_dropout=False,
#                                               perturb=True)
#         user_view_2, item_view_2 = self.gcn(self.user_embed,
#                                               self.item_embed,
#                                               edge_dropout=False,
#                                               mess_dropout=False,
#                                               perturb=True)
#         user_view_1, user_view_2, item_view_1, item_view_2 = self.pooling(user_view_1), self.pooling(user_view_2), self.pooling(item_view_1), self.pooling(item_view_2)
                  
#         user_cl_loss = self.InfoNCE(user_view_1[user_idx], user_view_2[user_idx], self.temperature)
#         item_cl_loss = self.InfoNCE(item_view_1[item_idx], item_view_2[item_idx], self.temperature)
#         return user_cl_loss + item_cl_loss

    def user_encoding(self, x):
        # i_emb = self.user_embed[x]
        i1_emb = self.dropout(x)
        i2_emb = self.dropout(x)

        i1_emb = self.user_tower(i1_emb)
        i2_emb = self.user_tower(i2_emb)

        return i1_emb, i2_emb
    
    def item_encoding(self, x):
        # i_emb = self.item_embed[x]
        i1_emb = self.dropout(x)
        i2_emb = self.dropout(x)

        i1_emb = self.item_tower(i1_emb)
        i2_emb = self.item_tower(i2_emb)

        return i1_emb, i2_emb
    
    def cal_cl_loss(self, user, item, temperature_u, temperature_i):
        user_view_1, user_view_2 = self.user_encoding(user)
        item_view_1, item_view_2 = self.item_encoding(item)   
        user_cl_loss, user_ = self.InfoNCE(user_view_1, user_view_2, temperature_u.unsqueeze(1))
        item_cl_loss, item_ = self.InfoNCE(item_view_1, item_view_2, temperature_i.unsqueeze(1))
        
        cl_loss = user_cl_loss + item_cl_loss
        
        return cl_loss, user_, item_


    def NO_Sample_Uniform_loss(self, user_gcn_emb, pos_gcn_emb, user, pos_item, w_0=None):
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
        
        
        if self.loss_name == "Adap_tau_Loss":
            mask_zeros = None
            
            tau = torch.index_select(self.memory_tau, 0, user).detach()
            tau_u = torch.index_select(self.memory_tau_u, 0, user).detach()
            tau_i = torch.index_select(self.memory_tau_i, 0, user).detach()
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            # Calculate contrastive loss
            # user_e, positive_e = self.gcn_emb()
            cl_loss, cl_loss_u, cl_loss_i = self.cal_cl_loss(u_e, pos_e, tau_u, tau_i)
            return loss.mean() + emb_loss + self.lamda * cl_loss, loss_, emb_loss, tau, self.lamda * cl_loss, cl_loss_u, cl_loss_i
        elif self.loss_name == "SSM_Loss":
            loss, loss_ = self.loss_fn(y_pred)
            return loss.mean() + emb_loss, loss_, emb_loss, y_pred
        else:
            raise NotImplementedError("loss={} is not support".format(self.loss_name))

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0

    # negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask