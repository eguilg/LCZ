import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_layers import *
from .cbam import CBAM_Module
from collections import OrderedDict


def densenet121(in_channel=20, num_classes=17, drop_rate=0, **kwargs):
	r"""Densenet-121 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZDenseNet(in_channel=in_channel, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
						drop_rate=drop_rate, n_class=num_classes,
						**kwargs)

	return model


def densenet169(in_channel=20, num_classes=17, drop_rate=0, **kwargs):
	r"""Densenet-169 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZDenseNet(in_channel=in_channel, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
						drop_rate=drop_rate, n_class=num_classes,
						**kwargs)

	return model


def densenet201(in_channel=20, num_classes=17, drop_rate=0, **kwargs):
	r"""Densenet-201 model from
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZDenseNet(in_channel=in_channel, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
						drop_rate=drop_rate, n_class=num_classes,
						**kwargs)

	return model


def densenet161(in_channel=20, num_classes=17, drop_rate=0, **kwargs):
	model = LCZDenseNet(in_channel=in_channel, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
						drop_rate=drop_rate, n_class=num_classes,
						**kwargs)

	return model


class _DenseLayer(nn.Sequential):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
		super(_DenseLayer, self).__init__()
		self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
		self.add_module('relu1', nn.ReLU(inplace=True)),
		self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
										   growth_rate, kernel_size=1, stride=1, bias=False)),
		self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),

		# self.add_module('cbma1', CBAM_Module(bn_size * growth_rate, 8))
		self.add_module('relu2', nn.ReLU(inplace=True)),
		self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
										   kernel_size=3, stride=1, padding=1, bias=False)),
		self.drop_rate = drop_rate

	def forward(self, x):
		new_features = super(_DenseLayer, self).forward(x)
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
		return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
	def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
			self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_Transition, self).__init__()
		self.add_module('norm', nn.BatchNorm2d(num_input_features))
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
										  kernel_size=1, stride=1, bias=False))
		self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class LCZDenseNet(nn.Module):

	def __init__(self, in_channel=3, n_class=17, growth_rate=32, block_config=(6, 12, 24, 16),
				 num_init_features=64, bn_size=4, drop_rate=0):

		super(LCZDenseNet, self).__init__()

		# First convolution
		self.features = nn.Sequential(OrderedDict([
			('conv0', nn.Conv2d(in_channel, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
			('norm0', nn.BatchNorm2d(num_init_features)),
			('relu0', nn.ReLU(inplace=True)),
			('pool0', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
		]))

		# Each denseblock
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
								bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

			self.features.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate

			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
				self.features.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

		# Final batch norm
		self.features.add_module('norm5', nn.BatchNorm2d(num_features))

		# Linear layer
		# self.classifier = nn.Linear(num_features, num_classes)
		# self.fc = nn.Conv2d(num_features, n_class, 4)
		self.fc = nn.Sequential(nn.Conv2d(num_features, n_class, 1),
								nn.AvgPool2d(4)
								)

		# Official init from torch repo.
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
		features = self.features(x)
		out = F.relu(features, inplace=True)
		# out = F.avg_pool2d(out, kernel_size=4, stride=1).view(features.size(0), -1)
		out = self.fc(out).view(out.size(0), -1)
		return out
