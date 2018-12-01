import torch
import torch.nn as nn
import torch.nn.functional as F


class PosAttn(nn.Module):
	"""
	位置注意力提取器
	"""

	def __init__(self, channel, base=32, dropout=0.2):
		super(PosAttn, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(channel, base, 3, 1, 1),
			nn.BatchNorm2d(base),
			nn.ReLU(),
			nn.Conv2d(base, base, 3, 1, 1),
			nn.BatchNorm2d(base),
			nn.ReLU(),
			nn.Dropout(0.2)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(base, 2 * base, 3, 1, 1),
			nn.BatchNorm2d(2 * base),
			nn.ReLU(),
			nn.Conv2d(2 * base, 2 * base, 3, 1, 1),
			nn.BatchNorm2d(2 * base),
			nn.ReLU(),
			nn.Dropout(0.3)
		)

		self.linear = nn.Linear(2 * base, 1)
		self.drop = nn.Dropout(dropout)

	def forward(self, input):  # b s s c
		input = input.transpose(2, 3).transpose(1, 2)
		attn = self.conv1(input)  # b base s s
		attn = self.conv2(attn)  # b 2base s s
		attn = attn.transpose(1, 2).transpose(2, 3)  # b s s 2base
		attn = F.sigmoid(self.linear(attn))  # b s s 1

		return attn


class ChannelAttn(nn.Module):
	"""
	通道注意力提取器
	"""

	def __init__(self, channel, base=64, dropout=0.2):
		super(ChannelAttn, self).__init__()
		self.drop = nn.Dropout(dropout)

		self.conv1 = nn.Sequential(
			nn.Conv2d(channel, base, 3, 1, 1),
			nn.BatchNorm2d(base),
			nn.ReLU(),
			nn.Conv2d(base, base, 3, 1, 1),
			nn.BatchNorm2d(base),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.2)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(base, 2 * base, 3, 1, 1),
			nn.BatchNorm2d(2 * base),
			nn.ReLU(),
			nn.Conv2d(2 * base, 2 * base, 3, 1, 1),
			nn.BatchNorm2d(2 * base),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.3)
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(2 * base, 4 * base, 3, 1, 1),
			nn.BatchNorm2d(4 * base),
			nn.ReLU(),
			nn.Conv2d(4 * base, 4 * base, 3, 1, 1),
			nn.BatchNorm2d(4 * base),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.4)
		)

		self.linear = nn.Linear(4 * base, channel)

		self.channel = channel

	def forward(self, input):
		input = input.transpose(2, 3).transpose(1, 2)
		f = self.conv1(input)  # b base 16 16
		f = self.conv2(f)  # b 2base 8 8
		f = self.conv3(f)  # b 4base 4 4
		f = F.max_pool2d(f, 2, 2)  # b 4base 2 2
		f = f.transpose(1, 2).transpose(2, 3)
		f = F.sigmoid(self.linear(f)).max(1, keepdim=True)[0].max(2, keepdim=True)[0]  # b 1 1 c
		return f


class LCZNet(nn.Module):
	"""
		LCZ分类网络
	"""

	def __init__(self, channel, n_class, base=64, dropout=0.2):
		super(LCZNet, self).__init__()
		self.n_class = n_class
		self.pos_att = PosAttn(channel, base, dropout)
		self.c_att = ChannelAttn(channel, base, dropout)
		self.drop = nn.Dropout(dropout)

		self.conv1 = nn.Sequential(
			nn.Conv2d(channel, base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(base),
			nn.Conv2d(base, base, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(base),
			nn.MaxPool2d(2, 2),
			nn.Dropout(0.2)
		)

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

		self.linear = nn.Linear(8 * base, n_class)
		self.channel = channel

	def forward(self, input):
		pos_att = self.pos_att(input)
		att_input = pos_att * input
		c_att = self.c_att(att_input)
		#  input with attention
		att_input = (c_att * att_input).transpose(2, 3).transpose(1, 2)
		out = self.conv1(att_input)  # b base 16 16
		out = self.conv2(out)  # b 2base 8 8
		out = self.conv3(out)  # b 4base 4 4
		out = self.conv4(out)  # b 8base 2 2
		out = out.transpose(1, 2).transpose(2, 3)  # b 2 2 4base
		out = F.softmax(self.linear(out).max(1)[0].max(1)[0], dim=-1)  # b class

		return out
