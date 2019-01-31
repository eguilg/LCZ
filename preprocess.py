import h5py
import torch
import math
from functools import partial
import numpy as np
from tqdm import tqdm
from dataloader import H5DataSource, MyDataLoader
from torchvision import transforms
from h5transform import *


def preprocess_batch(x_b):
	VH_ORG = x_b[:, :, :, :2]
	VH_ORG_A = torch.norm(VH_ORG, dim=-1)[:, :, :, None]
	VV_ORG = x_b[:, :, :, 2:4]
	VV_ORG_A = torch.norm(VV_ORG, dim=-1)[:, :, :, None]
	VV_M_VH_ORG = VH_ORG - VV_ORG
	VV_M_VH_ORG_A = torch.norm(VV_M_VH_ORG, dim=-1)[:, :, :, None]

	VH_LEE_A = torch.sqrt(x_b[:, :, :, 4:5])
	VV_LEE_A = torch.sqrt(x_b[:, :, :, 5:6])
	CO_RI = x_b[:, :, :, 6:8]
	CO_A = torch.norm(CO_RI, dim=-1)[:, :, :, None]
	S1_BANDS = torch.cat([
		1 - torch.exp(-VH_ORG_A),
		1 - torch.exp(-VV_ORG_A),
		1 - torch.exp(-VV_M_VH_ORG_A),
		1 - torch.exp(-VH_LEE_A),
		1 - torch.exp(-VV_LEE_A),
		1 - torch.exp(-CO_A),
	], dim=-1)
	S2_BANDS = x_b[:, :, :, 8:]

	# B8 & b4
	L = 0.428
	SAVI = (S2_BANDS[:, :, :, 6:7] - S2_BANDS[:, :, :, 2:3]) / (S2_BANDS[:, :, :, 6:7] + S2_BANDS[:, :, :, 2:3] + L) * (
			1.0 + L)  # -0.65 ~ 0.65
	SAVI = (SAVI + 0.65) / 1.3  # B8 & B4
	PSSR = (S2_BANDS[:, :, :, 6:7] / S2_BANDS[:, :, :, 2:3].clamp_min(1e-20) - 0.058) / (17.157 - 0.058)
	PSSR = PSSR.clamp(0, 1)  # b8 & b4
	NDVI = (S2_BANDS[:, :, :, 6:7] - S2_BANDS[:, :, :, 2:3]) / (
			S2_BANDS[:, :, :, 6:7] + S2_BANDS[:, :, :, 2:3]).clamp_min(1e-20)
	NDVI = (NDVI + 1) / 2  # b8 & b4
	UCNDVI = 1 - 2 * torch.sqrt(
		(S2_BANDS[:, :, :, 6:7] * 0.02) ** 2 + (S2_BANDS[:, :, :, 2:3] * 0.03) ** 2);  # uncertainty of NDVI

	# B8 & B11 / B12
	NDWI = (S2_BANDS[:, :, :, 6:7] - S2_BANDS[:, :, :, 8:9]) / (
			S2_BANDS[:, :, :, 6:7] + S2_BANDS[:, :, :, 8:9]).clamp_min(1e-20)
	NDWI = (NDWI + 1) / 2
	MSI = (S2_BANDS[:, :, :, 8:9] / S2_BANDS[:, :, :, 6:7].clamp_min(1e-20) - 0.058) / (17.145 - 0.058)
	MSI = MSI.clamp(0, 1)
	NBR = (S2_BANDS[:, :, :, 6:7] - S2_BANDS[:, :, :, 9:10]) / (
			S2_BANDS[:, :, :, 6:7] + S2_BANDS[:, :, :, 9:10]).clamp_min(1e-20)
	NBR = (NBR + 1) / 2

	# lower wave length
	MCARI = ((S2_BANDS[:, :, :, 3:4] - S2_BANDS[:, :, :, 2:3]) - 0.2 * (
			S2_BANDS[:, :, :, 3:4] - S2_BANDS[:, :, :, 1:2])) * (
					S2_BANDS[:, :, :, 3:4] / S2_BANDS[:, :, :, 1:2].clamp_min(1e-20));
	MCARI = ((MCARI - (-1.03)) / (4.606 + 1.03)).clamp(0, 1)
	GNDVI = (S2_BANDS[:, :, :, 6:7] - S2_BANDS[:, :, :, 1:2]) / (
			S2_BANDS[:, :, :, 6:7] + S2_BANDS[:, :, :, 1:2]).clamp_min(1e-20)
	GNDVI = (GNDVI + 1) / 2
	CHLRED = (S2_BANDS[:, :, :, 3:4] / S2_BANDS[:, :, :, 5:6].clamp_min(1e-20) - 0.058) / (17.149 - 0.058)
	CHLRED = CHLRED.clamp(0, 1)

	INDICE_BANDS = torch.cat([
		SAVI, PSSR, NDVI, UCNDVI, NDWI, MSI, NBR, MCARI, GNDVI, CHLRED
	], dim=-1)

	x_b = torch.cat([
		S1_BANDS,  # 0 6
		S2_BANDS,  # 1 10
		INDICE_BANDS  # 2 4, 3, 3
	], dim=-1)

	return x_b


def random_crop(img, pad=4):
	h, w, _ = img.shape

	img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

	i = np.random.randint(0, 2 * pad)
	j = np.random.randint(0, 2 * pad)

	img = img[i:i + h, j:j + w, :]
	return img


def random_crop_batch(batch_img, pad=4, mean=None):
	mapfunc = partial(random_crop, pad=pad, mean=mean)
	batch_img = np.array(list(map(mapfunc, batch_img)))
	return batch_img


def data_aug(x_b):
	batch_size = x_b.shape[0]

	#  flip h
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.5]
	x_b[random_idx] = np.flip(x_b[random_idx], 1)

	#  flip v
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.5]
	x_b[random_idx] = np.flip(x_b[random_idx], 2)

	#  transpose
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.5]
	x_b[random_idx] = np.transpose(x_b[random_idx], (0, 2, 1, 3))

	#  rotate 90
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.5]
	x_b[random_idx] = np.rot90(x_b[random_idx], 1, axes=(1, 2))

	#  rotate 180
	random_idx = np.arange(batch_size)[np.random.rand(batch_size) > 0.5]
	x_b[random_idx] = np.rot90(x_b[random_idx], 2, axes=(1, 2))

	# random crop
	x_b = random_crop_batch(x_b)

	return x_b


def data_aug_torch(x_b, trans):
	for i in range(x_b.shape[0]):
		x_b[i] = trans(x_b[i])
	return x_b


trans = transforms.Compose([
	H5RandomHorizontalFlip(cuda=True),
	H5RandomVerticalFlip(cuda=True),
	H5RandomRotate(cuda=True),
	H5RandomCrop(32, 4),
	Cutout(8, 0.5)
])


def prepare_batch(x_b, y_b, f_idx=None, mean=None, std=None, aug=False):
	x_b = torch.from_numpy(x_b).float().cuda()

	x_b = preprocess_batch(x_b)
	x_b = x_b.permute(0, 3, 1, 2)  # bs nc h w

	if aug:
		x_b = data_aug_torch(x_b, trans)

	if mean is not None and std is not None:
		if len(mean.shape) == 2 and f_idx is not None:
			mean = mean[f_idx]
			std = std[f_idx]
			x_b = (x_b - mean[:, None, None, :]) / std[:, None, None, :]
		else:
			x_b = (x_b - mean[None, :]) / std[None, :]

	if y_b is not None:
		y_b = torch.from_numpy(y_b).float().cuda()

	return x_b, y_b


def mixup_data(x, y, alpha=1.0):
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.shape[0]
	index = torch.randperm(batch_size).cuda()

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == '__main__':
	from config import *

	init_data_source = H5DataSource([test_file], 5000, shuffle=False, split=False)
	init_loader = MyDataLoader(init_data_source.h5fids, init_data_source.indices)
	mean, std, n = 0, 0, 0
	for data, label, _ in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)
		data = data.view(-1, data.size(3))
		mean += data.sum(0)
		n += data.shape[0]

	mean /= n

	for data, label, _ in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)
		data = data.view(-1, data.size(3))
		std += ((data - mean) ** 2).sum(0)

	std = torch.sqrt(std / n).detach().cpu().numpy()
	mean = mean.detach().cpu().numpy()
	print(mean)
	print(std)
	mean_std_h5 = h5py.File(mean_std_test_file, 'a')
	mean_std_h5.create_dataset('mean', data=mean)
	mean_std_h5.create_dataset('std', data=std)
	mean_std_h5.close()
