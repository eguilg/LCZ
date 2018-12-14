import h5py
import torch
import math
import numpy as np
from tqdm import tqdm
from dataloader import H5DataSource, MyDataLoader


def preprocess_batch(x_b, mean=None, std=None):
	VH_ORG = x_b[:, :, :, :2]
	VH_ORG_A = torch.norm(VH_ORG, dim=-1)[:, :, :, None]
	VH_ORG_SIGN = (VH_ORG[:, :, :, 1:2] > 0).float().float() * 2 - 1
	VH_ORG_PHI = VH_ORG_SIGN * (VH_ORG[:, :, :, 0:1] / (VH_ORG_A + 1e-20) + 1)

	VV_ORG = x_b[:, :, :, 2:4]
	VV_ORG_A = torch.norm(VV_ORG, dim=-1)[:, :, :, None]
	VV_ORG_SIGN = (VV_ORG[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_ORG_PHI = VV_ORG_SIGN * (VV_ORG[:, :, :, 0:1] / (VV_ORG_A + 1e-20) + 1)

	VV_VH_ORG = VH_ORG + VV_ORG * math.sqrt(1 / 2)
	VV_VH_ORG_A = torch.norm(VV_VH_ORG, dim=-1)[:, :, :, None]
	VV_VH_ORG_SIGN = (VV_VH_ORG[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_VH_ORG_PHI = VV_VH_ORG_SIGN * (VV_VH_ORG[:, :, :, 0:1] / (VV_VH_ORG_A + 1e-20) + 1)

	VV_M_VH_ORG = VH_ORG - VV_ORG * math.sqrt(1 / 2)
	VV_M_VH_ORG_A = torch.norm(VV_M_VH_ORG, dim=-1)[:, :, :, None]
	VV_M_VH_ORG_SIGN = (VV_M_VH_ORG[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_M_VH_ORG_PHI = VV_M_VH_ORG_SIGN * (VV_M_VH_ORG[:, :, :, 0:1] / (VV_M_VH_ORG_A + 1e-20) + 1)

	ORG_BANDS = torch.cat([
		1 - torch.exp(-math.sqrt(2) * VH_ORG_A),
		(VH_ORG_PHI + 2) / 4,
		1 - torch.exp(-VV_ORG_A),
		(VV_ORG_PHI + 2) / 4,
		1 - torch.exp(-VV_VH_ORG_A),
		(VV_VH_ORG_PHI + 2) / 4,
		1 - torch.exp(-VV_M_VH_ORG_A),
		(VV_M_VH_ORG_PHI + 2) / 4
	], dim=-1)

	VH_LEE_A = torch.sqrt(x_b[:, :, :, 4:5])
	VH_LEE_RI = VH_ORG * (VH_LEE_A / (VH_ORG_A + 1e-20))

	VV_LEE_A = torch.sqrt(x_b[:, :, :, 5:6])
	VV_LEE_RI = VV_ORG * (VV_LEE_A / (VV_ORG_A + 1e-20))

	VV_VH_LEE_RI = VH_LEE_RI + VV_LEE_RI * math.sqrt(1 / 2)
	VV_VH_LEE_A = torch.norm(VV_VH_LEE_RI, dim=-1)[:, :, :, None]
	VV_VH_LEE_SIGN = (VV_VH_LEE_RI[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_VH_LEE_PHI = VV_VH_LEE_SIGN * (VV_VH_LEE_RI[:, :, :, 0:1] / (VV_VH_LEE_A + 1e-20) + 1)

	VV_M_VH_LEE_RI = VH_LEE_RI - VV_LEE_RI * math.sqrt(1 / 2)
	VV_M_VH_LEE_A = torch.norm(VV_M_VH_LEE_RI, dim=-1)[:, :, :, None]
	VV_M_VH_LEE_SIGN = (VV_M_VH_LEE_RI[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_M_VH_LEE_PHI = VV_M_VH_LEE_SIGN * (VV_M_VH_LEE_RI[:, :, :, 0:1] / (VV_M_VH_LEE_A + 1e-20) + 1)

	CO_RI = x_b[:, :, :, 6:8]
	CO_A = torch.norm(CO_RI, dim=-1)[:, :, :, None]
	CO_SIGN = (CO_RI[:, :, :, 1:2] > 0).float() * 2 - 1
	CO_PHI = CO_SIGN * (CO_RI[:, :, :, 0:1] / (CO_A + 1e-20) + 1)

	LEE_BANDS = torch.cat([
		1 - torch.exp(-math.sqrt(2) * VH_LEE_A),
		1 - torch.exp(-VV_LEE_A),
		1 - torch.exp(-VV_VH_LEE_A),
		(VV_VH_LEE_PHI + 2) / 4,
		1 - torch.exp(-VV_M_VH_LEE_A),
		(VV_M_VH_LEE_PHI + 2) / 4,
		1 - torch.exp(-CO_A),
		(CO_PHI + 2) / 4
	], dim=-1)

	x_b = torch.cat([ORG_BANDS,  # 0 8
					 LEE_BANDS,  # 1 8
					 x_b[:, :, :, 8:]], dim=-1)

	if mean is not None and std is not None:
		x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
	return x_b


def preprocess_batch_org(x_b, mean=None, std=None):
	VH_ORG = x_b[:, :, :, :2]
	VH_ORG_A = torch.norm(VH_ORG, dim=-1)[:, :, :, None]
	VH_ORG_SIGN = (VH_ORG[:, :, :, 1:2] > 0).float().float() * 2 - 1
	VH_ORG_PHI = VH_ORG_SIGN * torch.sigmoid(VH_ORG[:, :, :, 1:2] / (VH_ORG[:, :, :, 0:1] + 1e-20))

	VV_ORG = x_b[:, :, :, 2:4]
	VV_ORG_A = torch.norm(VV_ORG, dim=-1)[:, :, :, None]
	VV_ORG_SIGN = (VV_ORG[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_ORG_PHI = VV_ORG_SIGN * torch.sigmoid(VV_ORG[:, :, :, 1:2] / (VV_ORG[:, :, :, 0:1] + 1e-20))

	VV_VH_ORG = VH_ORG + VV_ORG * math.sqrt(1 / 2)
	VV_VH_ORG_A = torch.norm(VV_VH_ORG, dim=-1)[:, :, :, None]
	VV_VH_ORG_SIGN = (VV_VH_ORG[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_VH_ORG_PHI = VV_VH_ORG_SIGN * torch.sigmoid(VV_VH_ORG[:, :, :, 1:2] / (VV_VH_ORG[:, :, :, 0:1] + 1e-20))

	VV_M_VH_ORG = VH_ORG - VV_ORG * math.sqrt(1 / 2)
	VV_M_VH_ORG_A = torch.norm(VV_M_VH_ORG, dim=-1)[:, :, :, None]
	VV_M_VH_ORG_SIGN = (VV_M_VH_ORG[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_M_VH_ORG_PHI = VV_M_VH_ORG_SIGN * torch.sigmoid(VV_M_VH_ORG[:, :, :, 1:2] / (VV_M_VH_ORG[:, :, :, 0:1] + 1e-20))

	ORG_BANDS = torch.cat([
		1 - torch.exp(-math.sqrt(2) * VH_ORG_A),
		(VH_ORG_PHI + 1) / 2,
		1 - torch.exp(-VV_ORG_A),
		(VV_ORG_PHI + 1) / 2,
		1 - torch.exp(-VV_VH_ORG_A),
		(VV_VH_ORG_PHI + 1) / 2,
		1 - torch.exp(-VV_M_VH_ORG_A),
		(VV_M_VH_ORG_PHI + 1) / 2
	], dim=-1)

	VH_LEE_A = torch.sqrt(x_b[:, :, :, 4:5])
	VH_LEE_RI = VH_ORG * (VH_LEE_A / (VH_ORG_A + 1e-20))

	VV_LEE_A = torch.sqrt(x_b[:, :, :, 5:6])
	VV_LEE_RI = VV_ORG * (VV_LEE_A / (VV_ORG_A + 1e-20))

	VV_VH_LEE_RI = VH_LEE_RI + VV_LEE_RI * math.sqrt(1 / 2)
	VV_VH_LEE_A = torch.norm(VV_VH_LEE_RI, dim=-1)[:, :, :, None]
	VV_VH_LEE_SIGN = (VV_VH_LEE_RI[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_VH_LEE_PHI = VV_VH_LEE_SIGN * torch.sigmoid(VV_VH_LEE_RI[:, :, :, 1:2] / (VV_VH_LEE_RI[:, :, :, 0:1] + 1e-20))

	VV_M_VH_LEE_RI = VH_LEE_RI - VV_LEE_RI * math.sqrt(1 / 2)
	VV_M_VH_LEE_A = torch.norm(VV_M_VH_LEE_RI, dim=-1)[:, :, :, None]
	VV_M_VH_LEE_SIGN = (VV_M_VH_LEE_RI[:, :, :, 1:2] > 0).float() * 2 - 1
	VV_M_VH_LEE_PHI = VV_M_VH_LEE_SIGN * torch.sigmoid(
		VV_M_VH_LEE_RI[:, :, :, 1:2] / (VV_M_VH_LEE_RI[:, :, :, 0:1] + 1e-20))

	CO_RI = x_b[:, :, :, 6:8]
	CO_A = torch.norm(CO_RI, dim=-1)[:, :, :, None]
	CO_SIGN = (CO_RI[:, :, :, 1:2] > 0).float() * 2 - 1
	CO_PHI = CO_SIGN * torch.sigmoid(CO_RI[:, :, :, 1:2] / (CO_RI[:, :, :, 0:1] + 1e-20))

	LEE_BANDS = torch.cat([
		1 - torch.exp(-math.sqrt(2) * VH_LEE_A),
		1 - torch.exp(-VV_LEE_A),
		1 - torch.exp(-VV_VH_LEE_A),
		(VV_VH_LEE_PHI + 1) / 2,
		1 - torch.exp(-VV_M_VH_LEE_A),
		(VV_M_VH_LEE_PHI + 1) / 2,
		1 - torch.exp(-CO_A),
		(CO_PHI + 1) / 2
	], dim=-1)

	x_b = torch.cat([ORG_BANDS,  # 0 8
					 LEE_BANDS,  # 1 8
					 x_b[:, :, :, 8:]], dim=-1)

	if mean is not None and std is not None:
		x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
	return x_b


def data_aug(x_b):
	batch_size = x_b.shape[0]

	#  flip h
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.75]
	x_b[random_idx] = np.flip(x_b[random_idx], 1)

	#  flip v
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.75]
	x_b[random_idx] = np.flip(x_b[random_idx], 2)

	#  transpose
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.75]
	x_b[random_idx] = np.transpose(x_b[random_idx], (0, 2, 1, 3))

	#  rotate 90
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.75]
	x_b[random_idx] = np.rot90(x_b[random_idx], 1, axes=(1, 2))

	#  rotate 180
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.75]
	x_b[random_idx] = np.rot90(x_b[random_idx], 2, axes=(1, 2))

	return x_b


def prepare_batch(x_b, y_b, f_idx=None, mean=None, std=None, aug=False):
	# x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
	if mean is not None and std is not None:
		mean = mean[f_idx]
		std = std[f_idx]
	if aug:
		x_b = data_aug(x_b)

	x_b = torch.from_numpy(x_b).float().cuda()

	x_b = preprocess_batch(x_b, mean, std)

	if y_b is not None:
		# y_b = torch.from_numpy(y_b).max(-1)[1].cuda()
		y_node_b = np.concatenate([y_b[:, :3].sum(-1).reshape(-1, 1),
								   y_b[:, 3:6].sum(-1).reshape(-1, 1),
								   y_b[:, 6:10].sum(-1).reshape(-1, 1),
								   y_b[:, 10:14].sum(-1).reshape(-1, 1),
								   y_b[:, 14:17].sum(-1).reshape(-1, 1)], -1)
		y_b = np.concatenate([y_node_b, y_b], -1)
		y_b = torch.from_numpy(y_b).long().cuda()
	return x_b[:, :, :, :], y_b


if __name__ == '__main__':
	train_file = '/home/zydq/Datasets/LCZ/training.h5'
	val_file = '/home/zydq/Datasets/LCZ/validation.h5'
	test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
	mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_train.h5'

	init_data_source = H5DataSource([train_file], 5000, shuffle=False, split=False)
	init_loader = MyDataLoader(init_data_source.h5fids, init_data_source.indices)
	mean, std, n = 0, 0, 0
	for data, label, _ in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)
		mean += data.sum(0)
		n += data.shape[0]

	mean /= n

	for data, label, _ in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)
		std += ((data - mean[None, :, :, :]) ** 2).sum(0)

	std = torch.sqrt(std / n).detach().cpu().numpy()
	mean = mean.detach().cpu().numpy()

	mean_std_h5 = h5py.File(mean_std_file, 'a')
	mean_std_h5.create_dataset('mean', data=mean)
	mean_std_h5.create_dataset('std', data=std)
	mean_std_h5.close()
