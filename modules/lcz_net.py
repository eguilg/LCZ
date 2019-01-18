import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_layers import *
from .cbam import CBAM_Module


class LCZNet(nn.Module):
	"""
		LCZ分类网络
	"""

	def __init__(self, channel, n_class, base=64, dropout=0.2):
		super(LCZNet, self).__init__()
		self.n_class = n_class

		self.conv1 = nn.Sequential(
			nn.Conv2d(channel, base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(base),
			nn.Conv2d(base, base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(base),
			nn.AvgPool2d(2, 2),
			nn.Dropout(0.2)
		)
		self.cbam1 = CBAM_Module(base, 8)
		self.conv2 = nn.Sequential(
			nn.Conv2d(base, 2 * base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(2 * base),
			nn.Conv2d(2 * base, 2 * base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(2 * base),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.3)
		)

		self.cbam2 = CBAM_Module(2 * base, 8)
		self.conv3 = nn.Sequential(
			nn.Conv2d(2 * base, 4 * base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(4 * base),
			nn.Conv2d(4 * base, 4 * base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(4 * base),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.4)
		)
		self.cbam3 = CBAM_Module(4 * base, 8)

		self.conv4 = nn.Sequential(
			nn.Conv2d(4 * base, 8 * base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(8 * base),
			nn.Conv2d(8 * base, 8 * base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(8 * base),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.4)
		)
		self.cbam4 = CBAM_Module(8 * base, 8)

		self.classifier = HierarchicalClassifier(8 * base, [3, 3, 4, 4, 3])
		self.channel = channel

	def forward(self, x):
		# x = x.transpose(2, 3).transpose(1, 2)
		f = self.cbam1(self.conv1(x))  # b base 16 16
		f = self.cbam2(self.conv2(f))  # b 2base 8 8
		f = self.cbam3(self.conv3(f))  # b 4base 4 4
		f = self.cbam4(self.conv4(f))  # b 8base 2 2
		f = F.max_pool2d(f, 2, 1)  # b 8base 1 1
		f = f.view(f.size(0), -1)
		node_out, leaf_out = self.classifier(f)
		return node_out, leaf_out
