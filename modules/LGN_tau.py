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
import faiss

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
        self.eps = 0.03

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
                mess_dropout=True, edge_dropout=True, perturbed=False):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]
        all_embeddings_cl = agg_embed
        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            if perturbed:
                random_noise = torch.rand_like(agg_embed).cuda()
                agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * self.eps
            embs.append(agg_embed)
            if hop==0:
                all_embeddings_cl = agg_embed
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        if perturbed:
            return embs[:self.n_users, :], embs[self.n_users:, :],all_embeddings_cl[:self.n_users, :], all_embeddings_cl[self.n_users:, :], embs
        return embs[:self.n_users, :], embs[self.n_users:, :], embs
    
class lgn_tau_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(lgn_tau_frame, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        
        self.k = self.n_users // 39
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.batch_size = args_config.batch_size
        self.cnt = args_config.cnt_lr
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
        self.register_buffer("memory_tau_u", torch.full((self.n_users,), 1 / 0.10))
        self.register_buffer("memory_tau_i", torch.full((self.n_users,), 1 / 0.10))
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        
        self.hyper_layers = 1
        self.ssl_reg = 1e-8
        self.alpha = 1.5
        self.proto_reg = 8e-7
        
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
        
        self.epoch = 0
        
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

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
    
    def _loss_to_tau1(self, x, x_all):
        if self.tau_mode == "weight_v0":
            t_0 = x_all
            tau = t_0 * torch.ones_like(self.memory_tau1, device=self.device)
        elif self.tau_mode == "weight_ratio":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau1, device=self.device)
            else:
                base_laberw = torch.quantile(x, self.temperature)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        elif self.tau_mode == "weight_mean":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(self.memory_tau1, device=self.device)
            else:
                base_laberw = torch.mean(x)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        return tau
    
    def cal_cl_loss(self,user_view1,user_view2,item_view1,item_view2,temperature_u, temperature_i):
        
        user_cl_loss, user_ = self.InfoNCE_1(user_view1, user_view2, temperature_u.unsqueeze(1))
        item_cl_loss, item_ = self.InfoNCE_1(item_view1, item_view2, temperature_i.unsqueeze(1))
        return user_cl_loss + item_cl_loss, user_, item_

    def forward(self, batch=None, loss_per_user=None, loss_per_user_u=None, loss_per_user_i=None, epoch=0, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        user_gcn_emb, item_gcn_emb, cl_user_emb, cl_item_emb, embs = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout, perturbed=True)
        self.epoch = epoch
        
        # if self.epoch >= 20:
        #     self.e_step()
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
            return self.NO_Sample_Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item],  cl_user_emb[user], cl_item_emb[pos_item], embs, user, pos_item, w_0)
        else:
            neg_item = batch['neg_items']
            return self.Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], item_gcn_emb[neg_item], user, w_0)
        
    
    
    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        # view1 = self.pooling(view1)
        # view2 = self.pooling(view2)
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        
        
        return -score.mean()
    
    def InfoNCE_1(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        # view1 = self.pooling(view1)
        # view2 = self.pooling(view2)
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        
        
        return -score.mean(), -score
    
       

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
        user_gcn_emb, item_gcn_emb, embs = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        return user_gcn_emb.detach(), item_gcn_emb.detach()
    
    def e_step(self):
        user_embeddings = self.user_embed.detach().cpu().numpy()
        item_embeddings = self.item_embed.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=False)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster
    
    def ProtoNCE_loss(self, node_embedding, user, item, temperature):
        user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        user_embeddings = user_embeddings_all[user]     # [B, e]
        # norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[user]     # [B,]
        user2centroids = self.user_centroids[user2cluster]   # [B, e]
#         pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
#         pos_score_user = torch.exp(pos_score_user / temperature.unsqueeze(1))
#         ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
#         ttl_score_user = torch.exp(ttl_score_user / temperature.unsqueeze(1)).sum(dim=1)

        
#         proto_nce_loss_user_ = -torch.log(pos_score_user / ttl_score_user)
        proto_nce_loss_user = self.InfoNCE(user_embeddings, user2centroids, temperature.unsqueeze(1)) * self.batch_size

        item_embeddings = item_embeddings_all[item]
        # norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.item_2cluster[item]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
#         pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
#         pos_score_item = torch.exp(pos_score_item / temperature.unsqueeze(1))
#         ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
#         ttl_score_item = torch.exp(ttl_score_item / temperature.unsqueeze(1)).sum(dim=1)
        
#         proto_nce_loss_item_ = -torch.log(pos_score_item / ttl_score_item)
        proto_nce_loss_item = self.InfoNCE(item_embeddings, item2centroids, temperature.unsqueeze(1)) * self.batch_size
        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def generate(self, mode='test', split=True):
        user_gcn_emb, item_gcn_emb, embs = self.gcn(self.user_embed,
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
        
    def ssl_layer_loss(self, context_emb, initial_emb, user, item, temperature):
        # print(context_emb.shape)
        context_user_emb_all, context_item_emb_all = torch.split(context_emb, [self.n_users, self.n_items])
        initial_user_emb_all, initial_item_emb_all = torch.split(initial_emb, [self.n_users, self.n_items])
        context_user_emb = context_user_emb_all[user]
        initial_user_emb = initial_user_emb_all[user]
        norm_user_emb1 = F.normalize(context_user_emb)
        norm_user_emb2 = F.normalize(initial_user_emb)
        norm_all_user_emb = F.normalize(initial_user_emb_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / temperature.unsqueeze(1))
        ttl_score_user = torch.exp(ttl_score_user / temperature.unsqueeze(1)).sum(dim=1)
        
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        
        
        context_item_emb = context_item_emb_all[item]
        initial_item_emb = initial_item_emb_all[item]
        norm_item_emb1 = F.normalize(context_item_emb)
        norm_item_emb2 = F.normalize(initial_item_emb)
        norm_all_item_emb = F.normalize(initial_item_emb_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / temperature.unsqueeze(1))
        ttl_score_item = torch.exp(ttl_score_item / temperature.unsqueeze(1)).sum(dim=1)
        
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()
       
        
        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

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


    def NO_Sample_Uniform_loss(self, user_gcn_emb, pos_gcn_emb, cl_user_emb, cl_item_emb, embs, user, pos_item, w_0=None):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb)  # [B, F]
        pos_e = self.pooling(pos_gcn_emb) # [B, F]
        emb_list = embs.transpose(0, 1)
        initial_emb = emb_list[0]
        context_emb = emb_list[2]
        
        proto_loss = 0
        ssl_user = 0
        proto_user = 0
        
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
            ssl_loss = self.ssl_layer_loss(context_emb,initial_emb, user, pos_item, tau)
            cl_loss, cl_loss_u, cl_loss_i = self.cal_cl_loss(u_e, cl_user_emb, pos_e, cl_item_emb, tau_u, tau_i)
            if self.epoch >= 20:
                proto_loss = self.ProtoNCE_loss(initial_emb, user, pos_item, tau)
        
                return loss.mean() + emb_loss + ssl_loss + proto_loss, loss_, emb_loss, tau, ssl_loss + proto_loss,  cl_loss_u, cl_loss_i
            else:
                return loss.mean() + emb_loss + ssl_loss + proto_loss, loss_ + ssl_user + proto_user, emb_loss, tau, ssl_loss + proto_loss, cl_loss_u, cl_loss_i
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