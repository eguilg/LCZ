import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)

def BasicConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0, bn=True):
	conv = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
	if bn:
		conv.append(nn.BatchNorm2d(out_channels))
	conv.append(nn.ReLU())
	if dropout > 0:
		conv.append(nn.Dropout(dropout))
	return nn.Sequential(*conv)


class HierarchicalClassifier(nn.Module):
	"""
	级联连分类器
	"""

	def __init__(self, feature_dim, leaf_nums):
		super(HierarchicalClassifier, self).__init__()
		self.node_classifier = nn.Linear(feature_dim, len(leaf_nums))
		self.leaf_classifiers = nn.ModuleList()
		for lf_n in leaf_nums:
			self.leaf_classifiers.append(nn.Linear(feature_dim, lf_n))

	def forward(self, input):
		node_pred = F.softmax(self.node_classifier(input), -1)
		leaf_preds = []
		for i, lf_clsfr in enumerate(self.leaf_classifiers):
			leaf_preds.append(F.softmax(lf_clsfr(input), -1) * node_pred[:, i][:, None])
		leaf_preds = torch.cat(leaf_preds, -1)
		return node_pred, leaf_preds