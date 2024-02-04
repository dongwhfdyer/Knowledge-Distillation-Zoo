from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
From https://github.com/lenscloth/RKD/blob/master/metric/loss.py
'''
class RKD(nn.Module):
	'''
	Relational Knowledge Distillation
	https://arxiv.org/pdf/1904.05068.pdf
	'''
	def __init__(self, w_dist, w_angle):
		super(RKD, self).__init__()

		self.w_dist  = w_dist
		self.w_angle = w_angle

	def forward(self, feat_s, feat_t):
		loss = self.w_dist * self.rkd_dist(feat_s, feat_t) + \
			   self.w_angle * self.rkd_angle(feat_s, feat_t)

		return loss

	def rkd_dist(self, feat_s, feat_t):
		feat_t_dist = self.pdist(feat_t, squared=False)
		mean_feat_t_dist = feat_t_dist[feat_t_dist>0].mean()
		feat_t_dist = feat_t_dist / mean_feat_t_dist

		feat_s_dist = self.pdist(feat_s, squared=False)
		mean_feat_s_dist = feat_s_dist[feat_s_dist>0].mean()
		feat_s_dist = feat_s_dist / mean_feat_s_dist

		loss = F.smooth_l1_loss(feat_s_dist, feat_t_dist)

		return loss

	def rkd_angle(self, feat_s, feat_t):
		# N x C --> N x N x C
		feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1))
		"""
		The line of code feat_t_vd = (feat_t.unsqueeze(0) - feat_t.unsqueeze(1)) in the RKD class of the
		 file kd_losses/rkd.py is creating a tensor feat_t_vd that contains the pairwise differences between the
		  vectors in the teacher feature map feat_t.

The unsqueeze() function is used to add a new dimension to the tensor along the specified axis. 
In this case, feat_t.unsqueeze(0) adds a new dimension at the beginning of the tensor, which has the effect of
 creating a new tensor with shape (1, N, C), where N is the number of vectors in the feature map 
 and C is the dimensionality of each vector. Similarly, feat_t.unsqueeze(1) adds a new dimension after the 
 first dimension, which has the effect of creating a new tensor with shape (N, 1, C).

Subtracting these two tensors creates a new tensor with shape (N, N, C), where each element (i, j, :) 
contains the difference between the vectors feat_t[i, :] and feat_t[j, :]. This tensor is then used to
 calculate the cosine similarity between the vectors in feat_t by normalizing it and taking the dot product 
 with its transpose, as explained in my previous response.
		"""
		norm_feat_t_vd = F.normalize(feat_t_vd, p=2, dim=2)
		feat_t_angle = torch.bmm(norm_feat_t_vd, norm_feat_t_vd.transpose(1, 2)).view(-1) # shape: (N,N, )

		feat_s_vd = (feat_s.unsqueeze(0) - feat_s.unsqueeze(1))
		norm_feat_s_vd = F.normalize(feat_s_vd, p=2, dim=2)
		feat_s_angle = torch.bmm(norm_feat_s_vd, norm_feat_s_vd.transpose(1, 2)).view(-1)

		loss = F.smooth_l1_loss(feat_s_angle, feat_t_angle)

		return loss

	def pdist(self, feat, squared=False, eps=1e-12):
		"""
		it calculates distance between each vector in the feature map.
		return feat_dist with shape of (feat_dim, feat_dim)
		"""
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0

		return feat_dist


