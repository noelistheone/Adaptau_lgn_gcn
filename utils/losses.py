import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class Adap_tau_Loss(nn.Module):
    def __init__(self):
        super(Adap_tau_Loss, self).__init__()

    def forward(self, y_pred, temperature_, w_0):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] *  w_0)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] * temperature_.unsqueeze(1))  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
        # log out
        pos_logits_ = torch.exp(y_pred[:, 0])  #  B
        neg_logits_ = torch.exp(y_pred[:, 1:])  # B M
        # neg_logits = torch.where(y_pred[:, 1:] > self._margin, neg_logits, mask_zeros)

        Ng_ = neg_logits_.sum(dim=-1)

        loss_ = (- torch.log(pos_logits_ / Ng_)).detach()
        return loss, loss_


    
class SSM_Loss(nn.Module):
    def __init__(self, margin=0, temperature=1.0, negative_weight=None, pos_mode=None):
        super(SSM_Loss, self).__init__()
        self._margin = margin 
        self._temperature = temperature
        self._negative_weight = negative_weight
        self.pos_mode = pos_mode
        print("Here is SSM LOSS: tau is {}".format(self._temperature))

    def forward(self, y_pred):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = torch.exp(y_pred[:, 0] / self._temperature)  #  B
        neg_logits = torch.exp(y_pred[:, 1:] / self._temperature)  # B M

        Ng = neg_logits.sum(dim=-1)

        loss = (- torch.log(pos_logits / Ng))
#         pos_scores = torch.mul(u_e, pos_e)
#         pos_scores = torch.sum(pos_scores, dim=1)
#         neg_scores = torch.mul(u_e, neg_e)
#         neg_scores = torch.sum(neg_scores, dim=1)
        
#         loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, 0.
    
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss
    

# class InfoNCE(nn.Module):
#     def _init_(self):
#         super(InfoNCE, self)._init_()
        
#     def forward(self, view1, view2, temperature: float, b_cos: bool = True):
#         """
#         Args:
#             view1: (torch.Tensor - N x D)
#             view2: (torch.Tensor - N x D)
#             temperature: float
#             b_cos (bool)

#         Return: Average InfoNCE Loss
#         """
#         # view1 = self.pooling(view1)
#         # view2 = self.pooling(view2)
#         if b_cos:
#             view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

#         pos_score = (view1 @ view2.T) / temperature
#         score = torch.diag(F.log_softmax(pos_score, dim=1))
#         # Compute scores without temperature scaling
#         pos_score_no_temp = (view1 @ view2.T)
#         score_no_temp = torch.diag(F.log_softmax(pos_score_no_temp, dim=1))
#         loss_ = -score_no_temp.detach()
        
#         return -score.mean(), loss_
    
# class InfoNCE(nn.Module):
#     def __init__(self):
#         super(InfoNCE, self).__init__()
        
#     def forward(self, views, temperature: float, b_cos: bool = True, dynamic_temp: bool = False, ohem: bool = True):
#         """
#         Args:
#             views: List of (torch.Tensor - N x D) multiple views
#             temperature: float
#             b_cos: bool
#             dynamic_temp: bool
#             ohem: bool

#         Return: Average InfoNCE Loss with and without temperature scaling
#         """
#         # Normalize views if b_cos is True
#         if b_cos:
#             views = [F.normalize(view, dim=1) for view in views]

#         # Get the number of views
#         num_views = len(views)
        
#         # Compute scores for each pair of views
#         all_scores = []
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 if dynamic_temp:
#                     # 动态温度调节，基于当前所有正负例的得分来调整温度
#                     temperature = self.adjust_temperature(views[i], views[j])
#                 score = (views[i] @ views[j].T) / temperature
#                 all_scores.append(score)
        
#         # Combine all scores
#         all_scores = torch.cat(all_scores, dim=1)

#         # 计算正例得分和负例得分
#         pos_scores = torch.diag(torch.cat([(views[i] @ views[i].T) / temperature for i in range(num_views)], dim=1))

#         if ohem:
#             # 在线难例挖掘
#             hard_neg_weights = self.ohem(all_scores, margin=0.2)
#             all_scores = all_scores * hard_neg_weights
        
#         # 计算带温度缩放的得分
#         loss_with_temp = -torch.diag(F.log_softmax(all_scores, dim=1)).mean()

#         # 计算不带温度缩放的得分
#         all_scores_no_temp = torch.cat([(views[i] @ views[j].T) for i in range(num_views) for j in range(i + 1, num_views)], dim=1)
#         loss_without_temp = -torch.diag(F.log_softmax(all_scores_no_temp, dim=1)).detach()
        
#         return loss_with_temp, loss_without_temp
    
#     def adjust_temperature(self, view1, view2):
#         """
#         动态调整温度参数，基于当前的得分
#         """
#         all_scores = view1 @ view2.T
#         mean_score = all_scores.mean().item()
#         new_temp = 1.0 / mean_score
#         return new_temp

#     def ohem(self, scores, margin):
#         """
#         在线难例挖掘函数
#         Args:
#             scores: 所有正负例的得分矩阵
#             margin: 在线难例挖掘的边距
#         """
#         positive_scores = torch.diag(scores)
#         hard_negatives = scores - positive_scores.unsqueeze(1) > margin
#         hard_neg_weights = hard_negatives.float() * 2.0 + 1.0
#         return hard_neg_weights

# class InfoNCE(nn.Module):
#     def __init__(self):
#         super(InfoNCE, self).__init__()
        

#     def forward(self, views, temperature: float, b_cos: bool = True, ohem: bool = True):
#         """
#         Args:
#             views: List of (torch.Tensor - N x D) multiple views
#             temperature: float
#             b_cos: bool
#             ohem: bool

#         Return: Average InfoNCE Loss with and without temperature scaling
#         """
#         # Normalize views if b_cos is True
#         if b_cos:
#             views = [F.normalize(view, dim=1) for view in views]

        

#         # Create the attention layer dynamically based on input dimension
#         self.attention_layer = nn.Sequential(
#             nn.Linear(64, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Tanh()
#         ).to(views[0].device)  # Ensure it is on the same device as the views

#         # Get the number of views
#         num_views = len(views)
        
#         # Compute scores for each pair of views
#         all_scores = []
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 score = (views[i] @ views[j].T) / temperature
#                 all_scores.append(score)
        
#         # Combine all scores
#         all_scores = torch.cat(all_scores, dim=1)

#         # Calculate positive scores
#         pos_scores = torch.diag(torch.cat([(views[i] @ views[i].T) / temperature for i in range(num_views)], dim=1))

#         if ohem:
#             # Online hard example mining
#             hard_neg_weights = self.ohem(all_scores, margin=0.2)
#             all_scores = all_scores * hard_neg_weights
        
#         # Calculate loss with temperature scaling
#         loss_with_temp = -torch.diag(F.log_softmax(all_scores, dim=1)).mean()

#         # Calculate loss without temperature scaling
#         all_scores_no_temp = torch.cat([(views[i] @ views[j].T) for i in range(num_views) for j in range(i + 1, num_views)], dim=1)
#         loss_without_temp = -torch.diag(F.log_softmax(all_scores_no_temp, dim=1)).detach()

#         # Attention Mechanism for View Importance
#         attention_weights = torch.cat([self.attention_layer(view) for view in views], dim=0)
#         attention_weights = F.softmax(attention_weights, dim=0).view(-1, 1)
#         views = [view * attention_weights[i] for i, view in enumerate(views)]

#         # Contrastive Cross-View Consistency Regularization
#         consistency_loss = 0.0
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 consistency_loss += F.mse_loss(views[i], views[j])
#         loss_with_temp += consistency_loss

#         return loss_with_temp, loss_without_temp

#     def ohem(self, scores, margin):
#         """
#         Online Hard Example Mining function
#         Args:
#             scores: Scores matrix of all positive and negative examples
#             margin: Margin for online hard example mining
#         """
#         positive_scores = torch.diag(scores)
#         hard_negatives = scores - positive_scores.unsqueeze(1) > margin
#         hard_neg_scores = scores[hard_negatives]
#         hard_neg_weights = torch.ones_like(scores)
#         hard_neg_weights[hard_negatives] = hard_neg_scores
#         return hard_neg_weights

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class InfoNCE(nn.Module):
#     def __init__(self):
#         super(InfoNCE, self).__init__()
        

#     def forward(self, views, temperature: float = None, b_cos: bool = True, ohem: bool = True):
#         """
#         Args:
#             views: List of (torch.Tensor - N x D) multiple views
#             temperature: float
#             b_cos: bool
#             ohem: bool

#         Return: Average InfoNCE Loss with and without temperature scaling
#         """
#         # Use the provided temperature or the learnable parameter
       

#         # Normalize views if b_cos is True
#         if b_cos:
#             views = [F.normalize(view, dim=1) for view in views]

#         # Extract multi-scale features
#         #views = self.multi_scale_features(views)

#         # Initialize the memory bank dynamically based on view dimensions
        

        

#         # Create the attention layer dynamically based on input dimension
#         self.attention_layer = nn.Sequential(
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Tanh()
#         ).to(views[0].device)  # Ensure it is on the same device as the views

#         # Get the number of views
#         num_views = len(views)
        
#         # Compute scores for each pair of views
#         all_scores = []
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 score = (views[i] @ views[j].T) / temperature
#                 all_scores.append(score)
        
#         # Combine all scores
#         all_scores = torch.cat(all_scores, dim=1)

#         # Calculate positive scores
#         pos_scores = torch.diag(torch.cat([(views[i] @ views[i].T) / temperature for i in range(num_views)], dim=1))

#         if ohem:
#             # Online hard example mining
#             hard_neg_weights = self.ohem(all_scores, margin=0.2)
#             all_scores = all_scores * hard_neg_weights
        
#         # Calculate loss with temperature scaling
#         loss_with_temp = -torch.diag(F.log_softmax(all_scores, dim=1)).mean()

#         # Calculate loss without temperature scaling
#         all_scores_no_temp = torch.cat([(views[i] @ views[j].T) for i in range(num_views) for j in range(i + 1, num_views)], dim=1)
#         loss_without_temp = -torch.diag(F.log_softmax(all_scores_no_temp, dim=1)).detach()

#         # Attention Mechanism for View Importance
#         attention_weights = torch.cat([self.attention_layer(view) for view in views], dim=0)
#         attention_weights = F.softmax(attention_weights, dim=0).view(-1, 1)
#         views = [view * attention_weights[i] for i, view in enumerate(views)]

#         # Momentum Contrastive Learning
        

#         # Contrastive Cross-View Consistency Regularization
#         consistency_loss = 0.0
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 consistency_loss += F.mse_loss(views[i], views[j])
#         loss_with_temp += consistency_loss

#         return loss_with_temp, loss_without_temp

#     def ohem(self, scores, margin):
#         """
#         Online Hard Example Mining function
#         Args:
#             scores: Scores matrix of all positive and negative examples
#             margin: Margin for online hard example mining
#         """
#         positive_scores = torch.diag(scores)
#         hard_negatives = scores - positive_scores.unsqueeze(1) > margin
#         hard_neg_scores = scores[hard_negatives]
#         hard_neg_weights = torch.ones_like(scores)
#         hard_neg_weights[hard_negatives] = hard_neg_scores
#         return hard_neg_weights

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class InfoNCE(nn.Module):
#     def __init__(self):
#         super(InfoNCE, self).__init__()
        

#     def forward(self, views, temperature: float = None, b_cos: bool = True, ohem: bool = True):
#         """
#         Args:
#             views: List of (torch.Tensor - N x D) multiple views
#             temperature: float
#             b_cos: bool
#             ohem: bool

#         Return: Average InfoNCE Loss with and without temperature scaling
#         """
#         if b_cos:
#             views = [F.normalize(view, dim=1) for view in views]

#         self.attention_layer = nn.Sequential(
#             nn.Linear(64, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Tanh()
#         ).to(views[0].device)

#         num_views = len(views)
        
#         all_scores = []
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 score = (views[i] @ views[j].T) / temperature
#                 all_scores.append(score)
        
#         all_scores = torch.cat(all_scores, dim=1)

#         pos_scores = torch.diag(torch.cat([(views[i] @ views[i].T) / temperature for i in range(num_views)], dim=1))

#         if ohem:
#             hard_neg_weights = self.ohem(all_scores, pos_scores, margin=0.2)
#             all_scores = all_scores * hard_neg_weights
        
#         loss_with_temp = -torch.diag(F.log_softmax(all_scores, dim=1)).mean()

#         all_scores_no_temp = torch.cat([(views[i] @ views[j].T) for i in range(num_views) for j in range(i + 1, num_views)], dim=1)
#         loss_without_temp = -torch.diag(F.log_softmax(all_scores_no_temp, dim=1)).detach()

#         attention_weights = torch.cat([self.attention_layer(view) for view in views], dim=0)
#         attention_weights = F.softmax(attention_weights, dim=0).view(-1, 1)
#         views = [view * attention_weights[i] for i, view in enumerate(views)]

#         consistency_loss = 0.0
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 consistency_loss += F.mse_loss(views[i], views[j])
#         loss_with_temp += consistency_loss

#         return loss_with_temp, loss_without_temp

#     def ohem(self, scores, pos_scores, margin):
#         """
#         Online Hard Example Mining function
#         Args:
#             scores: Scores matrix of all positive and negative examples
#             pos_scores: Positive scores for the examples
#             margin: Margin for online hard example mining
#         """
#         # Compute adaptive margin based on the distribution of positive scores
#         adaptive_margin = margin * torch.std(pos_scores).item()

#         positive_scores = torch.diag(scores)
#         hard_negatives = scores - positive_scores.unsqueeze(1) > adaptive_margin
#         hard_neg_scores = scores[hard_negatives]
        
#         # Use a more sophisticated weighting strategy
#         hard_neg_weights = torch.ones_like(scores)
#         if hard_neg_scores.numel() > 0:
#             weights = 1 + torch.sigmoid(hard_neg_scores - adaptive_margin)
#             hard_neg_weights[hard_negatives] = weights
        
#         return hard_neg_weights

class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()

    def forward(self, views, temperature, b_cos: bool = True, ohem: bool = True):
        """
        Args:
            views: List of (torch.Tensor - N x D) multiple views
            temperature: float
            b_cos: bool
            ohem: bool

        Return: Average InfoNCE Loss with and without temperature scaling
        """
        if b_cos:
            views = [F.normalize(view, dim=1) for view in views]

        num_views = len(views)
        
        # Symmetric InfoNCE
        all_scores = []
        all_scores_rev = []
        for i in range(num_views):
            for j in range(i + 1, num_views):
                score = (views[i] @ views[j].T) / temperature
                score_rev = (views[j] @ views[i].T) / temperature
                all_scores.append(score)
                all_scores_rev.append(score_rev)
        
        all_scores = torch.cat(all_scores, dim=1)
        all_scores_rev = torch.cat(all_scores_rev, dim=1)
        
        pos_scores = torch.diag(torch.cat([(views[i] @ views[i].T) / temperature for i in range(num_views)], dim=1))

        if ohem:
            hard_neg_weights = self.ohem(all_scores, pos_scores, margin=0.2)
            all_scores = all_scores * hard_neg_weights
            hard_neg_weights_rev = self.ohem(all_scores_rev, pos_scores, margin=0.2)
            all_scores_rev = all_scores_rev * hard_neg_weights_rev
        
        loss_with_temp = -torch.diag(F.log_softmax(all_scores, dim=1)).mean()
        loss_with_temp_rev = -torch.diag(F.log_softmax(all_scores_rev, dim=1)).mean()
        loss_with_temp = (loss_with_temp + loss_with_temp_rev) / 2  # Symmetric loss

        all_scores_no_temp = torch.cat([(views[i] @ views[j].T) for i in range(num_views) for j in range(i + 1, num_views)], dim=1)
        loss_without_temp = -torch.diag(F.log_softmax(all_scores_no_temp, dim=1)).detach()

        return loss_with_temp, loss_without_temp

    def ohem(self, scores, pos_scores, margin):
        """
        Online Hard Example Mining function
        Args:
            scores: Scores matrix of all positive and negative examples
            pos_scores: Positive scores for the examples
            margin: Margin for online hard example mining
        """
        adaptive_margin = margin * torch.std(pos_scores).item()
        positive_scores = torch.diag(scores)
        hard_negatives = scores - positive_scores.unsqueeze(1) > adaptive_margin
        hard_neg_scores = scores[hard_negatives]
        hard_neg_weights = torch.ones_like(scores)
        if hard_neg_scores.numel() > 0:
            weights = 1 + torch.sigmoid(hard_neg_scores - adaptive_margin)
            hard_neg_weights[hard_negatives] = weights
        return hard_neg_weights
    
# class DualContrastiveLoss(torch.nn.Module):
#     def __init__(self):
#         super(DualContrastiveLoss, self).__init__()
        

#     def forward(self, user_emb, pos_emb, temperature):
#         batch_size = user_emb.size(0)
#         # 计算相似度矩阵
#         sim_matrix = torch.matmul(user_emb, pos_emb.T) / temperature
#         # 取对角线元素作为正样本相似度
#         pos_sim = torch.diag(sim_matrix)
#         # 对比学习损失
#         labels = torch.arange(batch_size).cuda()
#         loss = F.cross_entropy(sim_matrix, labels)
#         return loss


    
class InfoNCE_m(nn.Module):
    def __init__(self):
        super(InfoNCE_m, self).__init__()
        self.momentum_encoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.momentum = 0.999

    def forward(self, views, temperature, b_cos: bool = True, ohem: bool = True):
        """
        Args:
            views: List of (torch.Tensor - N x D) multiple views
            temperature: float
            b_cos: bool
            ohem: bool

        Return: Average InfoNCE Loss with and without temperature scaling
        """
        if b_cos:
            views = [F.normalize(view, dim=1) for view in views]

        num_views = len(views)
        
        # Momentum Contrast (MoCo)
        with torch.no_grad():
            momentum_views = [self.momentum_encoder(view) for view in views]
            for param_q, param_k in zip(views, momentum_views):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        # Symmetric InfoNCE
        all_scores = []
        all_scores_rev = []
        for i in range(num_views):
            for j in range(i + 1, num_views):
                score = (views[i] @ views[j].T) / temperature
                score_rev = (momentum_views[j] @ momentum_views[i].T) / temperature
                all_scores.append(score)
                all_scores_rev.append(score_rev)
        
        all_scores = torch.cat(all_scores, dim=1)
        all_scores_rev = torch.cat(all_scores_rev, dim=1)
        
        pos_scores = torch.diag(torch.cat([(views[i] @ views[i].T) / temperature for i in range(num_views)], dim=1))

        if ohem:
            hard_neg_weights = self.ohem(all_scores, pos_scores, margin=0.2)
            hard_neg_weights_rev = self.ohem(all_scores_rev, pos_scores, margin=0.2)
            all_scores = all_scores * hard_neg_weights
            all_scores_rev = all_scores_rev * hard_neg_weights_rev
        
        loss_with_temp = -torch.diag(F.log_softmax(all_scores, dim=1)).mean()
        loss_with_temp_rev = -torch.diag(F.log_softmax(all_scores_rev, dim=1)).mean()
        loss_with_temp = (loss_with_temp + loss_with_temp_rev) / 2  # Symmetric loss

        all_scores_no_temp = torch.cat([(views[i] @ views[j].T) for i in range(num_views) for j in range(i + 1, num_views)], dim=1)
        loss_without_temp = -torch.diag(F.log_softmax(all_scores_no_temp, dim=1)).detach()

        return loss_with_temp, loss_without_temp

    def ohem(self, scores, pos_scores, margin):
        """
        Online Hard Example Mining function
        Args:
            scores: Scores matrix of all positive and negative examples
            pos_scores: Positive scores for the examples
            margin: Margin for online hard example mining
        """
        adaptive_margin = margin * torch.std(pos_scores).item()
        positive_scores = torch.diag(scores)
        hard_negatives = scores - positive_scores.unsqueeze(1) > adaptive_margin
        hard_neg_scores = scores[hard_negatives]
        hard_neg_weights = torch.ones_like(scores)
        if hard_neg_scores.numel() > 0:
            weights = 1 + torch.sigmoid(hard_neg_scores - adaptive_margin)
            hard_neg_weights[hard_negatives] = weights
        return hard_neg_weights