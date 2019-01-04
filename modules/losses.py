import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss


class FocalCE(nn.Module):

	def __init__(self, lam=2, weight=None, size_average=None,
				 reduce=None, reduction='mean'):
		super(FocalCE, self).__init__()
		self.lam = lam
		self.ce = torch.nn.CrossEntropyLoss(weight, size_average,
											reduce, reduction)

	def forward(self, input, target):

		foc = torch.pow(torch.abs(input - target), self.lam)

		focal_loss = foc * self.ce(input, target.max(-1)[1])

		return focal_loss
