import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_layers import *
from .cbam import CBAM_Module


def CBAM_Conv(in_c, out_c, kernel, stride, padding, reduction=8, dropout=0.0, bn=True):
	conv = [BasicConv(in_c, out_c, kernel, 1, padding, bn=bn),
			BasicConv(out_c, out_c, kernel, stride, padding, bn=bn),
			CBAM_Module(out_c, reduction),
			]
	if dropout > 0:
		conv.append(nn.Dropout(dropout))
	return nn.Sequential(*conv)


def AlignConv(in_c, out_c, kernel, stride, padding, dropout=0.0, bn=True):
	conv = [BasicConv(in_c, out_c, kernel, 1, padding, bn=bn),
			BasicConv(out_c, out_c, kernel, stride, padding, bn=bn)
			]
	if dropout > 0:
		conv.append(nn.Dropout(dropout))
	return nn.Sequential(*conv)


class GACNet(nn.Module):

	def __init__(self, group_sizes, class_nodes, base_size, dropout=0.3, bn=True):
		super(GACNet, self).__init__()
		self.group_sizes = group_sizes
		self.class_nodes = class_nodes
		self.base_size = base_size

		self.groups = nn.ModuleList()
		group_feature_dim = 0
		for group_size in group_sizes:
			n_fileter = base_size * max(group_size, 2)
			self.groups.append(CBAM_Conv(group_size, n_fileter, 3, 2, 1, n_fileter // 2, 0.5, bn))
			# self.groups.append(AlignConv(group_size, n_fileter, 3, 2, 1, 0.5, bn))
			group_feature_dim += n_fileter

		self.bottelneck = nn.Sequential(
										AlignConv(group_feature_dim, 512, 3, 2, 1, 0.3, bn=bn),
										CBAM_Conv(512, 1024, 3, 2, 1, 8, 0.3, bn=bn),
										# AlignConv(512, 1024, 3, 2, 1, 0.3, bn=bn)
										)

		# self.classifier = nn.Linear(1024, 17)
		self.classifier = HierarchicalClassifier(1024, class_nodes)

	def forward(self, x):

		x = x.transpose(2, 3).transpose(1, 2)  # b channel s s

		cur_c = 0
		out = []
		for i, group_size in enumerate(self.group_sizes):
			out.append(self.groups[i](x[:, cur_c: cur_c + group_size, :, :]))
			cur_c += group_size

		out = torch.cat(out, dim=1)  # b, c, s/2, s/2

		out = self.bottelneck(out)  # b, 16*base, s/4, s/4

		out = F.max_pool2d(out, 4, stride=1)
		out = out.view(out.size(0), -1)

		node_out, out = self.classifier(out)
		return node_out, out
		# out = F.softmax(self.classifier(out), dim=-1)
		# return out
