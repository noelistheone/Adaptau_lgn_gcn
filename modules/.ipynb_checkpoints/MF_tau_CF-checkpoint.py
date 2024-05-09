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

class MF_tau_cf(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(MF_tau_cf, self).__init__()

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
        # init  setting
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.loss_name = args_config.loss_fn
        self.generate_mode = args_config.generate_mode
        self.reg_loss = losses.L2Loss()
        self.lamda = 50#########################################################超参数定义
        self.eps = 0.1
        
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
        # define loss function
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
        
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        
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
        
    def add_noise(self, user_embed, item_embed,
        mess_dropout=True, edge_dropout=True, perturb = False):
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        #print(all_embed)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.context_hops):
            interact_mat = self.sparse_dropout(self.sparse_norm_adj,
                                                    self.edge_dropout_rate) if edge_dropout \
                                                                            else self.sparse_norm_adj

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
    

    def _update_tau_memory(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau = x

    # def _loss_to_tau(self, x, x_all):
    #     t_0 = x_all
    #     if x is None:
    #         tau = t_0 * torch.ones_like(self.memory_tau, device=self.device)
    #     else:
    #         base_laberw = torch.mean(x)
    #         laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
    #                                 min=-np.e ** (-1), max=1000)
    #         laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
    #         tau = (t_0 * torch.exp(-laberw_data)).detach()
    #     return tau
    
    
    
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
    
    def reg_data_handler(self, user, item, user_emb, item_emb):
        #print(user_embed.shape)
        #print(item_embed.shape)
        # u_online, i_online = self.add_noise(self.user_embed,
        #                              self.item_embed,
        #                              edge_dropout=False,
        #                              mess_dropout=False,
        #                              perturb=True)
        u_online, i_online = self.user_embed, self.item_embed
        with torch.no_grad():
            u_target, i_target = u_online.clone(), i_online.clone()
            
            # edge pruning
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

        u_online = u_online[user, :]
        i_online = i_online[item, :]
        return u_online, u_target, i_online, i_target
    
    def loss_cf(self, p, z):  # negative cosine similarity
        p = p.view(-1, p.size(0))
        p = torch.transpose(p, 0, 1)
        #print(p.shape)
        #print(z.shape)
        # print(p.t().shape)
        # print(z.detach().shape)

        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    
    def calculate_cf_loss(self, u_online, u_target, i_online, i_target):
        #reg_loss = self.reg_loss(u_online, i_online)

        u_online_1, i_online_1 = self.predictor(u_online), self.predictor(i_online)
        
        loss_ui = self.loss_cf(u_online_1, i_target)/2
        loss_iu = self.loss_cf(i_online_1, u_target)/2
        # loss_uu = self.loss_cf(u_online_1, u_target)/4
        # loss_ii = self.loss_cf(i_online_1, i_target)/4

        return loss_ui + loss_iu
    
    def forward(self, batch=None, loss_per_user=None, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        u_online, u_target, i_online, i_target = self.reg_data_handler(user, pos_item, self.user_embed[user], self.item_embed[pos_item])
        if s == 0 and w_0 is not None:
            tau_user = self._loss_to_tau(loss_per_user, w_0)
            self._update_tau_memory(tau_user)
         
        return self.Uniform_loss(self.user_embed[user], self.item_embed[pos_item], self.item_embed[neg_item], user, u_online, u_target, i_online, i_target, w_0)

    def gcn_emb(self):
        user_gcn_emb, item_gcn_emb = self.user_embed, self.item_embed
        # user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        return user_gcn_emb.detach(), item_gcn_emb.detach()
    
    def generate(self, mode='test', split=True):
        user_gcn_emb = self.user_embed
        item_gcn_emb = self.item_embed
        if self.generate_mode == "cosine":
            if self.u_norm:
                user_gcn_emb = F.normalize(user_gcn_emb, dim=-1)
            if self.i_norm:
                item_gcn_emb = F.normalize(item_gcn_emb, dim=-1)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # 对比训练loss，仅仅计算角度
    def Uniform_loss(self, user_gcn_emb, pos_gcn_emb, neg_gcn_emb, user, u_online, u_target, i_online, i_target, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = user_gcn_emb  # [B, F]
        if self.mess_dropout:
            u_e = self.dropout(u_e)
        pos_e = pos_gcn_emb # [B, F]
        neg_e = neg_gcn_emb # [B, M, F]

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
        cf_loss = self.lamda * self.calculate_cf_loss(u_online, u_target, i_online, i_target)
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
