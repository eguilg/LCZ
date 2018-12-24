import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_layers import *
from .cbam import CBAM_Module

from torchvision.models.resnet import BasicBlock, Bottleneck


class CbamBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(CbamBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)

		self.cbam = CBAM_Module(planes, 8)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out = self.cbam(out)
		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class CbamBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(CbamBottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		self.cbam = CBAM_Module(planes * 4, 16)


	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		out = self.cbam(out)
		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class LCZResNet(nn.Module):

	def __init__(self, block, in_planes, layers, class_nodes=None, num_classes=17):
		self.inplanes = 64
		super(LCZResNet, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=3, stride=1, padding=1,
							   bias=False)  # 32 * 32
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 16 * 16
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 8 * 8
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 4 * 4
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 2 * 2
		self.avgpool = nn.AvgPool2d(2, stride=1)
		# self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.fc = HierarchicalClassifier(512 * block.expansion, class_nodes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = x.transpose(2, 3).transpose(1, 2)  # b channel s s

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x_nodes, x = self.fc(x)

		return x_nodes, x


def resnet10(in_planes=20, class_nodes=None, **kwargs):
	"""Constructs a ResNet-10 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZResNet(CbamBlock, in_planes, [1, 1, 1, 1], class_nodes, **kwargs)

	return model


def resnet18(in_planes=20, class_nodes=None, **kwargs):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZResNet(CbamBlock, in_planes, [2, 2, 2, 2], class_nodes, **kwargs)

	return model


def resnet34(in_planes=20, class_nodes=None, **kwargs):
	"""Constructs a ResNet-34 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZResNet(CbamBlock, in_planes, [3, 4, 6, 3], class_nodes, **kwargs)

	return model


def resnet50(in_planes=20, class_nodes=None, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZResNet(CbamBottleneck, in_planes, [3, 4, 6, 3], class_nodes, **kwargs)

	return model


def resnet101(in_planes=20, class_nodes=None, **kwargs):
	"""Constructs a ResNet-101 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZResNet(CbamBottleneck, in_planes, [3, 4, 23, 3], class_nodes, **kwargs)

	return model


def resnet152(in_planes=20, class_nodes=None, **kwargs):
	"""Constructs a ResNet-152 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = LCZResNet(CbamBottleneck, in_planes, [3, 8, 36, 3], class_nodes, **kwargs)

	return model
