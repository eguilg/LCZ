import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_layers import *
from .cbam import CBAM_Module


def CBAM_Conv(in_c, out_c, kernel, stride, padding, reduction=8, dropout=0.0, pool=None, bn=True):
	conv = [BasicConv(in_c, out_c, kernel, 1, padding, bn=bn),
			BasicConv(out_c, out_c, kernel, stride, padding, bn=bn),
			CBAM_Module(out_c, reduction),
			]
	if pool is not None:
		conv.append(pool)
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

	def __init__(self, group_sizes, n_class, base_size, dropout=0.3, bn=True):
		super(GACNet, self).__init__()
		self.group_sizes = group_sizes
		self.n_class = n_class
		self.base_size = base_size

		self.groups = nn.ModuleList()
		group_feature_dim = 0
		for group_size in group_sizes:
			n_fileter = base_size * min(group_size, 3)
			self.groups.append(
				CBAM_Conv(group_size, n_fileter,
						  kernel=3, stride=1, padding=1,
						  reduction=2, dropout=0.5,
						  pool=nn.MaxPool2d(2, 2),
						  bn=bn))  # 16 * 16
			group_feature_dim += n_fileter

		self.bottelneck = nn.Sequential(
			CBAM_Conv(group_feature_dim, 512,
					  kernel=3, stride=1, padding=1,
					  reduction=8, dropout=0.3,
					  pool=nn.MaxPool2d(2, 2), bn=bn),  # 8 * 8
			CBAM_Conv(512, 1024,
					  kernel=3, stride=1, padding=1,
					  reduction=8, dropout=0.3,
					  pool=nn.MaxPool2d(2, 2), bn=bn),  # 4 * 4


		)

		self.fc = nn.Sequential(nn.Conv2d(1024, n_class, 1),
								nn.AvgPool2d(4)
								)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def forward(self, x):

		x = x.transpose(2, 3).transpose(1, 2)  # b channel s s

		cur_c = 0
		out = []
		for i, group_size in enumerate(self.group_sizes):
			out.append(self.groups[i](x[:, cur_c: cur_c + group_size, :, :]))
			cur_c += group_size

		out = torch.cat(out, dim=1)  # b, c, s/2, s/2

		out = self.bottelneck(out)  # b, 16*base, s/4, s/4

		out = F.softmax(self.fc(out).view(out.size(0), -1), -1)

		return out

