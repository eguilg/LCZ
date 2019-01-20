import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


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

		nll = -(target * F.log_softmax(input, -1) + (1 - target) * torch.log(1 - F.softmax(input, -1)))
		focal_loss = (sample_w * (foc * nll).sum(-1)).mean()

		return focal_loss


class GHMC_Loss(nn.Module):
	def __init__(self, bins=10, momentum=0.9):
		super(GHMC_Loss, self).__init__()
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
		weights = torch.zeros(input.size(0)).cuda()

		# gradient length
		g = (torch.abs(torch.softmax(input, -1).detach() - target) * target).sum(-1)


		tot = max(input.size(0), 1.0)
		n = 0  # n valid bins
		for i in range(self.bins):
			inds = (g >= edges[i]) & (g < edges[i + 1])
			num_in_bin = inds.sum().item()
			if num_in_bin > 0:
				if mmt > 0:
					self.acc_sum[i] = mmt * self.acc_sum[i] \
									  + (1 - mmt) * num_in_bin
					weights[inds] = tot / self.acc_sum[i]
				else:
					weights[inds] = tot / num_in_bin
				n += 1
		if n > 0:
			weights = weights / n
		loss = (weights * F.cross_entropy(input, target.max(-1)[1], reduction='none')).mean()
		return loss
