import h5py
import torch
import numpy as np
from tqdm import tqdm
from dataloader import H5DataSource, MyDataLoader


def preprocess_batch(x_b, mean=None, std=None):
	VH_ORG = x_b[:, :, :, :2]
	VH_ORG_A = torch.norm(VH_ORG, dim=-1)
	VH_ORG_IN = VH_ORG_A[:, :, :, None] ** 2  # 0

	VV_ORG = x_b[:, :, :, 2:4]
	VV_ORG_A = torch.norm(VV_ORG, dim=-1)
	VV_ORG_IN = VV_ORG_A[:, :, :, None] ** 2  # 1

	VV_VH_ORG = VH_ORG + VV_ORG * np.sqrt(1 / 2)
	VV_VH_ORG_IN = torch.norm(VV_VH_ORG, dim=-1)[:, :, :, None] ** 2  # 2

	VV_M_VH_ORG = VH_ORG - VV_ORG * np.sqrt(1 / 2)
	VV_M_VH_ORG_IN = torch.norm(VV_M_VH_ORG, dim=-1)[:, :, :, None] ** 2  # 3

	VH_LEE1_IN = x_b[:, :, :, 4][:, :, :, None]  # 4
	VH_LEE_A = torch.sqrt(x_b[:, :, :, 4])
	VV_LEE1_IN = x_b[:, :, :, 5][:, :, :, None]  # 5
	VV_LEE_A = torch.sqrt(x_b[:, :, :, 5])
	Rc = x_b[:, :, :, 6][:, :, :, None]

	VV_VH_LEE1_IN = VH_LEE1_IN + VV_LEE1_IN / 2 + Rc  # 6
	VV_M_VH_LEE1_IN = VH_LEE1_IN + VV_LEE1_IN / 2 - Rc  # 7

	VV_LEE = VV_ORG * (VV_LEE_A / (VV_ORG_A + 1e-20))[:, :, :, None]
	VH_LEE = VH_ORG * (VH_LEE_A / (VH_ORG_A + 1e-20))[:, :, :, None]

	VV_VH_LEE2 = VV_LEE + VH_LEE * np.sqrt(1 / 2)
	VV_M_VH_LEE2 = VV_LEE - VH_LEE * np.sqrt(1 / 2)

	VV_VH_LEE2_IN = torch.norm(VV_VH_LEE2, dim=-1)[:, :, :, None]  # 8
	VV_M_VH_LEE2_IN = torch.norm(VV_M_VH_LEE2, dim=-1)[:, :, :, None]  # 9

	x_b = torch.cat([1 - torch.exp(-2 * VH_ORG_IN),
					 1 - torch.exp(-VV_ORG_IN),
					 1 - torch.exp(-VV_VH_ORG_IN),
					 1 - torch.exp(-VV_M_VH_ORG_IN),
					 1 - torch.exp(-2 * VH_LEE1_IN),
					 1 - torch.exp(-VV_LEE1_IN),
					 1 - torch.exp(-VV_VH_LEE1_IN),
					 1 - torch.exp(-VV_M_VH_LEE1_IN),
					 1 - torch.exp(-VV_VH_LEE2_IN),
					 1 - torch.exp(-VV_M_VH_LEE2_IN),
					 x_b[:, :, :, 8:]], dim=-1)

	if mean is not None and std is not None:
		x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
	return x_b


def preprocess_batch_1(x_b, mean=None, std=None):
	VH_ORG = x_b[:, :, :, :2]
	VH_ORG_A = torch.norm(VH_ORG, dim=-1)[:, :, :, None]
	# VH_ORG_IN = VH_ORG_A ** 2
	VH = torch.cat([VH_ORG, VH_ORG_A], dim=-1)  # 0 ,3

	VV_ORG = x_b[:, :, :, 2:4]
	VV_ORG_A = torch.norm(VV_ORG, dim=-1)[:, :, :, None]
	# VV_ORG_IN = VV_ORG_A ** 2
	VV = torch.cat([VV_ORG, VV_ORG_A], dim=-1)  # 1 ,3

	VV_VH_ORG = VH_ORG + VV_ORG * np.sqrt(1 / 2)
	VV_VH_ORG_A = torch.norm(VV_VH_ORG, dim=-1)[:, :, :, None]
	VV_VH = torch.cat([VV_VH_ORG, VV_VH_ORG_A], dim=-1)  # 2, 3

	VV_M_VH_ORG = VH_ORG - VV_ORG * np.sqrt(1 / 2)
	VV_M_VH_ORG_A = torch.norm(VV_M_VH_ORG, dim=-1)[:, :, :, None]
	VV_M_VH = torch.cat([VV_M_VH_ORG, VV_M_VH_ORG_A], dim=-1)  # 3, 3

	VH_LEE_A = torch.sqrt(x_b[:, :, :, 4:5])
	VH_LEE_RI = VH_ORG * (VH_LEE_A / (VH_ORG_A + 1e-20))
	VH_LEE = torch.cat([VH_LEE_RI, VH_LEE_A], dim=-1)  # 4, 3

	VV_LEE_A = torch.sqrt(x_b[:, :, :, 5:6])
	VV_LEE_RI = VV_ORG * (VV_LEE_A / (VV_ORG_A + 1e-20))
	VV_LEE = torch.cat([VV_LEE_RI, VV_LEE_A], dim=-1)  # 5, 3

	CO_RI = x_b[:, :, :, 6:8]
	CO_A = torch.norm(CO_RI, dim=-1)[:, :, :, None]
	CO = torch.cat([CO_RI, CO_A], dim=-1)  # 6, 3

	VH_LEE_IN = x_b[:, :, :, 4:5]
	VV_LEE_IN = x_b[:, :, :, 5:6]
	Rc = x_b[:, :, :, 6:7]

	VV_VH_LEE1_IN = VH_LEE_IN + VV_LEE_IN / 2 + Rc
	VV_VH_LEE1_A = torch.sqrt(VV_VH_LEE1_IN)
	VV_M_VH_LEE1_IN = VH_LEE_IN + VV_LEE_IN / 2 - Rc
	VV_M_VH_LEE1_A = torch.sqrt(VV_M_VH_LEE1_IN)

	VV_VH_LEE_RI = VV_LEE_RI + VH_LEE_RI * np.sqrt(1 / 2)
	VV_VH_LEE_A = torch.norm(VV_VH_LEE_RI, dim=-1)[:, :, :, None]
	VV_VH_LEE = torch.cat([VV_VH_LEE_RI, VV_VH_LEE_A, VV_VH_LEE1_A], dim=-1)  # 7, 4

	VV_M_VH_LEE_RI = VV_LEE_RI - VH_LEE_RI * np.sqrt(1 / 2)
	VV_M_VH_LEE_A = torch.norm(VV_VH_LEE_RI, dim=-1)[:, :, :, None]
	VV_M_VH_LEE = torch.cat([VV_M_VH_LEE_RI, VV_M_VH_LEE_A, VV_M_VH_LEE1_A], dim=-1)  # 8, 4

	x_b = torch.cat([VH, VV, VV_VH, VV_M_VH,
					 VH_LEE, VV_LEE, CO,
					 VV_VH_LEE, VV_M_VH_LEE,
					 x_b[:, :, :, 8:]], dim=-1)

	if mean is not None and std is not None:
		x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
	return x_b


def data_aug_1(x_b):
	batch_size = x_b.size(0)

	#  flip v
	random_idx = torch.arange(batch_size).masked_select(torch.rand(batch_size) > 0.5).long()
	x_b[random_idx] = x_b[random_idx].flip(1)

	# flip h
	random_idx = torch.arange(batch_size).masked_select(torch.rand(batch_size) > 0.5).long()

	x_b[random_idx] = x_b[random_idx].flip(2)

	# transpose
	random_idx = torch.arange(batch_size).masked_select(torch.rand(batch_size) > 0.5).long()
	x_b[random_idx] = x_b[random_idx].transpose(1, 2)

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


def prepare_batch(x_b, y_b, mean=None, std=None, aug=False):
	# x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]

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
	mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_trainval.h5'

	init_data_source = H5DataSource([train_file, val_file], 5000, shuffle=False, split=False)
	init_loader = MyDataLoader(init_data_source.h5fids, init_data_source.indices)
	mean, std, n = 0, 0, 0
	for data, label in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)
		mean += data.sum(0)
		n += data.shape[0]

	mean /= n

	for data, label in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)
		std += ((data - mean[None, :, :, :]) ** 2).sum(0)

	std = torch.sqrt(std / n).detach().cpu().numpy()
	mean = mean.detach().cpu().numpy()

	mean_std_h5 = h5py.File(mean_std_file, 'a')
	mean_std_h5.create_dataset('mean', data=mean)
	mean_std_h5.create_dataset('std', data=std)
	mean_std_h5.close()
