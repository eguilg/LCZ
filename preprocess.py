import h5py
import torch
import numpy as np
from tqdm import tqdm
from dataloader import H5DataSource, MyDataLoader

train_file = '/home/zydq/Datasets/LCZ/training.h5'
val_file = '/home/zydq/Datasets/LCZ/validation.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_trainval.h5'


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


def prepare_batch(x_b, y_b, mean=None, std=None):
	# x_b = (x_b - mean[None, None, None, :]) / std[None, None, None, :]
	# x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
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
