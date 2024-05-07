import math
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class NTXentLoss(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0,
                 covariance_reg=0) -> None:
        super(NTXentLoss, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def std_loss(self,x):
        std = torch.sqrt(x.var(dim=0) + 1e-04)
        return torch.mean(torch.relu(1 - std))

    def cov_loss(self,x):
        batch_size, metric_dim = x.size()
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (batch_size - 1)
        off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
        return off_diag_cov.pow_(2).sum() / metric_dim + 1e-8

    def uniformity_loss(self,x1,x2,t=2):
        sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
        uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
        sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
        uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
        return (uniformity_x1 + uniformity_x2) / 2

    def forward(self, z1, z2, **kwargs):
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim+ 1e-8)
        loss=loss.float()
        loss = - torch.log(loss+ 1e-8).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (self.std_loss(z1) + self.std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (self.cov_loss(z1) + self.cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * self.uniformity_loss(z1, z2)
        return loss

