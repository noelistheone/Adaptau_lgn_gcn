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
    
class InfoNCE(nn.Module):
    def _init_(self):
        super(InfoNCE, self)._init_()
        
    def forward(self, view1, view2, temperature: float, b_cos: bool = True):
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
        # Compute scores without temperature scaling
        pos_score_no_temp = (view1 @ view2.T)
        score_no_temp = torch.diag(F.log_softmax(pos_score_no_temp, dim=1))
        loss_ = -score_no_temp.detach()
        
        return -score.mean(), loss_
