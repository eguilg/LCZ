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


