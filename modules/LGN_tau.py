
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


class AdaptiveWeighting:
    def __init__(self, device, epsilon=1e-8, smooth_factor=0.05, min_weight=0.05):
        self.device = device
        self.epsilon = epsilon
        self.smooth_factor = smooth_factor
        self.min_weight = min_weight
        self.initial_losses = None
        self.moving_avg_losses = None
        self.task_weights = None

    def update_losses_and_weights(self, loss, cl_loss, cl_loss_s):
        # Convert losses to tensors and ensure they are on the correct device
        current_losses = torch.tensor([loss.mean().item(), cl_loss.item(), cl_loss_s.item()], device=self.device)

        # Initialize initial_losses and moving_avg_losses on the first iteration
        if self.initial_losses is None:
            self.initial_losses = current_losses.clone()
            self.moving_avg_losses = current_losses.clone()

        # Normalize current losses
        normalized_losses = current_losses / (self.moving_avg_losses + self.epsilon)

        # Compute task difficulties based on moving average differences
        task_difficulties = current_losses / (self.moving_avg_losses + self.epsilon)

        # Update moving averages
        self.moving_avg_losses = 0.9 * self.moving_avg_losses + 0.1 * current_losses

        # Update task weights based on difficulties
        exp_difficulties = torch.exp(-task_difficulties)  # Use negative to give less weight to more difficult tasks
        raw_task_weights = exp_difficulties / (exp_difficulties.sum() + self.epsilon)
        
        # Apply smoothing to task weights
        if self.task_weights is None:
            self.task_weights = raw_task_weights
        else:
            self.task_weights = (1 - self.smooth_factor) * self.task_weights + self.smooth_factor * raw_task_weights

        # Ensure each task weight is above a minimum threshold
        self.task_weights = torch.max(self.task_weights, torch.tensor([self.min_weight], device=self.device).expand_as(self.task_weights))

        # Compute final weighted loss
        weighted_loss = (self.task_weights[0] * loss +
                         self.task_weights[1] * cl_loss +
                         self.task_weights[2] * cl_loss_s).sum()

        return weighted_loss

# class GraphConv(nn.Module):
#     """
#     Graph Convolutional Network
#     """
#     def __init__(self, n_hops, n_users, interact_mat,
#                  edge_dropout_rate=0.5, mess_dropout_rate=0.1):
#         super(GraphConv, self).__init__()

#         self.interact_mat = interact_mat
#         self.n_users = n_users
#         self.n_hops = n_hops
#         self.edge_dropout_rate = edge_dropout_rate
#         self.mess_dropout_rate = mess_dropout_rate
        

#         self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

#     def _sparse_dropout(self, x, rate=0.5):
#         noise_shape = x._nnz()

#         random_tensor = rate
#         random_tensor += torch.rand(noise_shape).to(x.device)
#         dropout_mask = torch.floor(random_tensor).type(torch.bool)
#         i = x._indices()
#         v = x._values()

#         i = i[:, dropout_mask]
#         v = v[dropout_mask]

#         out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
#         return out * (1. / (1 - rate))
    
    

#     def forward(self, user_embed, item_embed,
#                 mess_dropout=True, edge_dropout=True, perturbed=False, eps=0.03):
#         # user_embed: [n_users, channel]
#         # item_embed: [n_items, channel]

#         # all_embed: [n_users+n_items, channel]
#         all_embed = torch.cat([user_embed, item_embed], dim=0)
#         agg_embed = all_embed
#         embs = [all_embed]
#         all_embeddings_cl = agg_embed
#         for hop in range(self.n_hops):
#             interact_mat = self._sparse_dropout(self.interact_mat,
#                                                 self.edge_dropout_rate) if edge_dropout \
#                                                                         else self.interact_mat

#             agg_embed = torch.sparse.mm(interact_mat, agg_embed)
#             if mess_dropout:
#                 agg_embed = self.dropout(agg_embed)
#             # agg_embed = F.normalize(agg_embed)
#             if perturbed:
#                 random_noise = torch.rand_like(agg_embed).cuda()
#                 agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * eps
#             embs.append(agg_embed)
#             if hop==0:
#                 all_embeddings_cl = agg_embed
#         embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
#         if perturbed:
#             return embs[:self.n_users, :], embs[self.n_users:, :],all_embeddings_cl[:self.n_users, :], all_embeddings_cl[self.n_users:, :]
#         return embs[:self.n_users, :], embs[self.n_users:, :]

class GraphConv(nn.Module):
    """
    Graph Convolutional Network with enhancements
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        # Initialize dropout layers and layer normalization
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.layer_norm = nn.LayerNorm(interact_mat.shape[1])

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate + torch.rand(noise_shape, device=x.device)
        dropout_mask = torch.floor(random_tensor).bool()
        i = x._indices()[:, dropout_mask]
        v = x._values()[dropout_mask]

        return torch.sparse.FloatTensor(i, v, x.shape, device=x.device) * (1. / (1 - rate))

    def forward(self, user_embed, item_embed, mess_dropout=True, edge_dropout=True, perturbed=False, eps=0.03):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # Combine user and item embeddings
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]
        all_embeddings_cl = agg_embed

        for hop in range(self.n_hops):
            # Apply sparse dropout on the interaction matrix
            interact_mat = self._sparse_dropout(self.interact_mat, self.edge_dropout_rate) if edge_dropout else self.interact_mat

            # Perform sparse matrix multiplication
            agg_embed = torch.sparse.mm(interact_mat, agg_embed)

            # Apply layer normalization
            agg_embed = self.layer_norm(agg_embed)

            # Apply message dropout if enabled
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            
            # Add perturbation noise if enabled
            if perturbed:
                random_noise = torch.randn_like(agg_embed, device=agg_embed.device)
                agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * eps

            embs.append(agg_embed)
            if hop == 0:
                all_embeddings_cl = agg_embed

        # Stack embeddings
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]

        if perturbed:
            return embs[:self.n_users, :], embs[self.n_users:, :], all_embeddings_cl[:self.n_users, :], all_embeddings_cl[self.n_users:, :]

        return embs[:self.n_users, :], embs[self.n_users:, :]

class lgn_tau_frame(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, logger=None):
        super(lgn_tau_frame, self).__init__()

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
        
        self.min_eps = 0.01
        self.max_eps = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.eps = torch.tensor([0.03], requires_grad=True).cuda()
        self.m_t = torch.zeros_like(self.eps).cuda()
        self.v_t = torch.zeros_like(self.eps).cuda()
        self.t = 0
        self.eps_grad = 0  # Placeholder for gradient of epsilon
        
        self.initial_losses = None
        # Initialize weights for adaptive loss weighting
        self.task_weights = torch.ones(3).to(self.device)  # Assuming 3 loss components: main loss, cl_loss, cl_loss_s
        
        self.smooth_factor = 0.1
        self.min_weight = 0.1
        
        self.moving_avg_losses = None
        

        # Initialize memory for storing historical losses
        
        self.epsilon = 1e-8  # For numerical stability
        # self.beta = 0.1  # Default value for weight adjustment
        
        # Define variants for SoftAdapt algorithm
        self.variants = ['Normalized', 'Loss Weighted']  # Example variants, can be modified
       
        # param for norm
        self.u_norm = args_config.u_norm
        self.i_norm = args_config.i_norm
        self.tau_mode = args_config.tau_mode
       
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
       
        self.loss_name = args_config.loss_fn
    
        self.generate_mode = args_config.generate_mode
        
        
        self.InfoNCE = losses.InfoNCE()
        self.weight = AdaptiveWeighting(self.device)
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
        self.register_buffer("memory_tau_us", torch.full((self.n_users,), 1 / 0.10))
        self.register_buffer("memory_tau_is", torch.full((self.n_users,), 1 / 0.10))
        self.gcn = self._init_model()
        self.sampling_method = args_config.sampling_method
        
        self.user_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.emb_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64)
        )
        
        self.dropout_rate = 0.1
        self.dropout = nn.Dropout(self.dropout_rate)

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
            
    def update_eps(self, meta_loss_grad):
        self.t += 1
        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * meta_loss_grad
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (meta_loss_grad ** 2)

        m_hat = self.m_t / (1 - self.beta1 ** self.t)
        v_hat = self.v_t / (1 - self.beta2 ** self.t)

        self.eps = self.eps - self.alpha * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        self.eps = self.eps.clamp_(self.min_eps, self.max_eps)
        
        """
        Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
        https://arxiv.org/pdf/1703.03400
        """
        # self.eps_grad = meta_loss_grad
        # self.eps = max(self.min_eps, min(self.max_eps, self.eps - self.beta * self.eps_grad))
        # return self.eps
    
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
    
    def _update_tau_memory_us(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau_us = x

    def _update_tau_memory_is(self, x):
        # x: std [B]
        # y: update position [B]
        with torch.no_grad():
            x = x.detach()
            self.memory_tau_is = x
    
    def _loss_to_tau(self, x, x_all, memory_tau):
        if self.tau_mode == "weight_v0":
            t_0 = x_all
            tau = t_0 * torch.ones_like(memory_tau, device=self.device)
        elif self.tau_mode == "weight_ratio":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(memory_tau, device=self.device)
            else:
                base_laberw = torch.quantile(x, self.temperature)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        elif self.tau_mode == "weight_mean":
            t_0 = x_all
            if x is None:
                tau = t_0 * torch.ones_like(memory_tau, device=self.device)
            else:
                base_laberw = torch.mean(x)
                laberw_data = torch.clamp((x - base_laberw) / self.temperature_2,
                                        min=-np.e ** (-1), max=1000)
                laberw_data = self.lambertw_table[((laberw_data + 1) * 1e4).long()]
                tau = (t_0 * torch.exp(-laberw_data)).detach()
        return tau
    
    def cal_cl_loss(self,user_view1,user_view2,item_view1,item_view2,temperature_u, temperature_i):
        
        user_cl_loss, user_ = self.InfoNCE([user_view1, user_view2], temperature_u.unsqueeze(1))
        item_cl_loss, item_ = self.InfoNCE([item_view1, item_view2], temperature_i.unsqueeze(1))
        
        return (user_cl_loss + item_cl_loss) * 1, user_, item_
    


    def forward(self, batch=None, loss_per_user=None, loss_per_user_u = None, loss_per_user_i=None, loss_per_user_us = None, loss_per_user_is=None,meta_loss_grad=None, epoch=None, w_0=None, s=0):
        user = batch['users']
        pos_item = batch['pos_items']
        if meta_loss_grad is not None:
            self.update_eps(meta_loss_grad)
        
        user_gcn_emb, item_gcn_emb, cl_user_emb, cl_item_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout, perturbed=True, eps=self.eps)
        
        # neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        if s == 0 and w_0 is not None:
            # self.logger.info("Start to adjust tau with respect to users")
            tau_user = self._loss_to_tau(loss_per_user, w_0, self.memory_tau)
            tau_user_u = self._loss_to_tau(loss_per_user_u, w_0, self.memory_tau_u)
            tau_user_i = self._loss_to_tau(loss_per_user_i, w_0, self.memory_tau_i)
            tau_user_us = self._loss_to_tau(loss_per_user_us, w_0, self.memory_tau_us)
            tau_user_is = self._loss_to_tau(loss_per_user_is, w_0, self.memory_tau_is)
            self._update_tau_memory(tau_user)
            self._update_tau_memory_u(tau_user_u)
            self._update_tau_memory_i(tau_user_i)
            self._update_tau_memory_us(tau_user_us)
            self._update_tau_memory_is(tau_user_is)
        if self.sampling_method == "no_sample":
            return self.NO_Sample_Uniform_loss(user_gcn_emb[user], item_gcn_emb[pos_item], cl_user_emb[user], cl_item_emb[pos_item], user, pos_item, epoch, w_0)
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
    
    def cal_cl_loss_Large_scale_Item(self, user, item, temperature_u, temperature_i):
        
        user_view_1, user_view_2 = self.user_encoding(user)
        user_view_3, user_view_4 = self.user_encoding(user)
        item_view_1, item_view_2 = self.item_encoding(item)   
        item_view_3, item_view_4 = self.item_encoding(item)   
        
        user_cl_loss, user_ = self.InfoNCE([user_view_1, user_view_2,user_view_3, user_view_4], temperature_u.unsqueeze(1))
        item_cl_loss, item_ = self.InfoNCE([item_view_1, item_view_2, item_view_3, item_view_4], temperature_i.unsqueeze(1))
        
        cl_loss = (user_cl_loss + item_cl_loss) * 1
        
        return cl_loss, user_, item_

    
    def NO_Sample_Uniform_loss(self, user_gcn_emb, pos_gcn_emb, cl_user_emb, cl_item_emb, user, pos_item, epoch, w_0=None):
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
            tau_us = torch.index_select(self.memory_tau_us, 0, user).detach()
            tau_is = torch.index_select(self.memory_tau_is, 0, user).detach()
            loss, loss_ = self.loss_fn(y_pred, tau, w_0)
            cl_loss, cl_loss_u, cl_loss_i = self.cal_cl_loss(u_e, cl_user_emb, pos_e, cl_item_emb, tau_u, tau_i)
            ########################
            ##############
            #########
            #这里要改一下
            cl_loss_s, cl_loss_us, cl_loss_is = self.cal_cl_loss_Large_scale_Item(u_e, pos_e, tau_us, tau_is)
            
            """
            SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks with Multi-Part Loss Functions
            https://ar5iv.labs.arxiv.org/html/1912.12355
            https://paperswithcode.com/paper/softadapt-techniques-for-adaptive-loss
            """

            """
            Multi-task Network Embedding with AdaptiveLoss Weighting
            https://dl.acm.org/doi/epdf/10.1109/ASONAM49781.2020.9381423
            """
            weighted_loss = self.weight.update_losses_and_weights(loss, cl_loss, cl_loss_s)
            
#             current_losses = torch.tensor([loss.mean().item(), cl_loss.item(), cl_loss_s.item()], device=self.device)

#             if self.initial_losses is None:
#                 # Initialize initial_losses on the first iteration
#                 self.initial_losses = current_losses.clone()

#             # Compute task difficulties
#             task_difficulties = current_losses / (self.initial_losses + self.epsilon)

#             # Update task weights based on difficulties
#             exp_difficulties = torch.exp(task_difficulties)
#             self.task_weights = exp_difficulties / (exp_difficulties.sum() + self.epsilon)
            
            

#             # Compute final weighted loss
#             weighted_loss = (self.task_weights[0] * loss.mean() +
#                              self.task_weights[1] * cl_loss +
#                              self.task_weights[2] * cl_loss_s)

            
            
            return weighted_loss + emb_loss, loss_, emb_loss, tau, cl_loss,cl_loss_u,cl_loss_i, cl_loss_us, cl_loss_is
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
