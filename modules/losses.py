import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss

import math

def cross_entropy_loss(input, target, reduction='mean'):

	input = F.log_softmax(input, dim=1)
	loss = -input * target
	if reduction == 'none':
		return loss
	elif reduction == 'mean':
		return loss.mean()
	elif reduction == 'sum':
		return loss.sum()
	else:
		raise ValueError('\'reduction\' must be one of \'none\', \'mean\' or \'sum\'')


class SoftCE(_WeightedLoss):
	__constants__ = ['weight', 'reduction']

	def __init__(self, lam=2, weight=None, size_average=None,
				 reduce=None, reduction='mean'):
		super(SoftCE, self).__init__(weight, size_average, reduce, reduction)
		self.lam = lam

	def forward(self, input, target):
		if self.weight is not None:
			sample_w = (target * self.weight[None, :]).sum(dim=-1)
		else:
			sample_w = 1

		label_entropy = 1 - (target * torch.log(target.clamp(min=1e-20))).sum(1) / math.log(1 / target.size(1))
		label_entropy = torch.pow(label_entropy, self.lam)

		ce = cross_entropy_loss(input, target, reduction='none')
		ce = (sample_w * label_entropy * ce.sum(-1)).mean()

		return ce


class FocalCE(_WeightedLoss):
	__constants__ = ['weight', 'reduction']

	def __init__(self, lam=2, weight=None, size_average=None,
				 reduce=None, reduction='mean'):
		super(FocalCE, self).__init__(weight, size_average, reduce, reduction)
		self.lam = lam

	def forward(self, input, target):
		if self.weight is not None:
			sample_w = (target * self.weight[None, :]).sum(dim=-1)
		else:
			sample_w = 1

		foc = torch.pow(torch.abs(F.softmax(input, -1) - target), self.lam)
		# foc = (torch.abs(F.softmax(input, -1) - target) * self.lam).exp() - 1

		ce = cross_entropy_loss(input, target, reduction='none')
		focal_loss = (sample_w * (foc * ce).sum(-1)).mean()

		return focal_loss


class GHMC_Loss(_Loss):

	def __init__(self, num_class=17, bins=30, momentum=0):
		super(GHMC_Loss, self).__init__()
		self.num_class = num_class
		self.bins = bins
		self.momentum = momentum
		self.edges = [float(x) / bins for x in range(bins + 1)]
		self.edges[-1] += 1e-6
		if momentum > 0:
			# self.acc_sum = [[0.0 for _ in range(bins)] for _ in range(num_class)]
			acc_sum = torch.zeros(bins, num_class)
			# acc_sum.fill_(1 / (bins * num_class))
			self.register_buffer('acc_sum', acc_sum)

	def forward(self, input, target):
		""" Args:
		input [batch_num, class_num]:
			The direct prediction of classification fc layer.
		target [batch_num, class_num]:
			Binary target (0 or 1) for each sample each class. The value is -1
			when the sample is ignored.
		"""
		edges = self.edges
		mmt = self.momentum

		# gradient length
		g = (torch.abs(torch.softmax(input, -1).detach() - target) * target).sum(-1)  # bs
		beta = torch.zeros(input.size(0)).cuda()

		tot = torch.mv(target, target.sum(0))
		epsilon = 1 / self.bins
		n = 0  # n valid bins
		for i in range(self.bins):
			inds = (g >= edges[i]) & (g < edges[i + 1])  # bs
			num_in_bin_by_c = torch.mm(inds.float().unsqueeze(0), target).squeeze(0)  # c
			if (num_in_bin_by_c > 0).sum() > 0:
				if mmt > 0:
					if self.training:
						self.acc_sum[i] = mmt * self.acc_sum[i] + \
										  (1 - mmt) * num_in_bin_by_c
					beta[inds] = epsilon / torch.mv(target[inds], self.acc_sum[i])
				# n += torch.mv(target, (self.acc_sum[i] > 0).float())
				else:
					beta[inds] = epsilon * tot[inds] / torch.mv(target[inds], num_in_bin_by_c)
				n += torch.mv(target, (num_in_bin_by_c > 0).float())
		if mmt > 0:
			beta *= torch.mv(target, self.acc_sum.sum(0))

		beta_expect = epsilon * n
		weights = beta / beta_expect
		loss = (weights * F.cross_entropy(input, target.max(-1)[1], reduction='none')).mean()
		return loss


class GHMC_Loss_ORG(_Loss):
	def __init__(self, bins=30, momentum=0.9):
		super(GHMC_Loss_ORG, self).__init__()
		self.bins = bins
		self.momentum = momentum
		self.edges = [float(x) / bins for x in range(bins + 1)]
		self.edges[-1] += 1e-6
		if momentum > 0:
			self.acc_sum = [0.0 for _ in range(bins)]

	def forward(self, input, target):
		""" Args:
		input [batch_num, class_num]:
			The direct prediction of classification fc layer.
		target [batch_num, class_num]:
			Binary target (0 or 1) for each sample each class. The value is -1
			when the sample is ignored.
		"""
		edges = self.edges
		mmt = self.momentum
		beta = torch.zeros(input.size(0)).cuda()

		# gradient length
		g = (torch.abs(torch.softmax(input, -1).detach() - target) * target).sum(-1)
		epsilon = 1 / self.bins
		tot = max(input.size(0), 1.0)
		n = 0  # n valid bins
		for i in range(self.bins):
			inds = (g >= edges[i]) & (g < edges[i + 1])
			num_in_bin = inds.sum().item()
			if num_in_bin > 0:
				if mmt > 0:
					if self.training:
						self.acc_sum[i] = mmt * self.acc_sum[i] \
										  + (1 - mmt) * num_in_bin
					beta[inds] = epsilon * tot / self.acc_sum[i]
				else:
					beta[inds] = epsilon * tot / num_in_bin
				n += 1

		beta_expect = epsilon * max(n, 1)
		weights = beta / beta_expect

		loss = (weights * F.cross_entropy(input, target.max(-1)[1], reduction='none')).mean()
		return loss
