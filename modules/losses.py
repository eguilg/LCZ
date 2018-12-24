import torch
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

		foc = torch.pow(torch.abs(input - target), self.lam)
		n_class = target.size(-1)
		nll_loss = - (target * torch.log(input.clamp(1e-20, 1)) + (1 - target) * torch.log((1 - input).clamp(1e-20, 1)))
		focal_loss = (sample_w * (foc * nll_loss).sum(-1) / n_class).mean()

		return focal_loss


class SoftCE(_WeightedLoss):
	__constants__ = ['weight', 'reduction']

	def __init__(self, weight=None, size_average=None,
				 reduce=None, reduction='mean'):
		super(SoftCE, self).__init__(weight, size_average, reduce, reduction)

	def forward(self, input, target):
		if self.weight is not None:
			sample_w = (target * self.weight[None, :]).sum(dim=-1)
		else:
			sample_w = 1

		n_class = target.size(-1)
		nll_loss = - (target * torch.log(input.clamp(1e-20, 1)) + (1 - target) * torch.log((1 - input).clamp(1e-20, 1)))
		loss = (sample_w * nll_loss.sum(-1) / n_class).mean()

		return loss
