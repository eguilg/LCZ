import torch  # for speed
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataloader import H5DataSource, MyDataLoader
from preprocess import preprocess_batch

NC_IN = 26
BATCH_SIZE = 3000


def get_dense_column(n_channel):
	column = ['mean_' + str(i) for i in range(n_channel)] + \
			 ['min_' + str(i) for i in range(n_channel)] + \
			 ['max_' + str(i) for i in range(n_channel)] + \
			 ['mid_' + str(i) for i in range(n_channel)] + \
			 ['std_' + str(i) for i in range(n_channel)] + \
			 ['per' + str(j) + '_' + str(i) for i in range(n_channel) for j in [10, 20, 40, 60, 80, 90]]
	return column


def GaborFilters(ksize=None, n_direct=6):
	filters = []
	if ksize is None:
		ksize = [3, 5, 7]
	for K in range(len(ksize)):
		filters.append([])
		for th_id, theta in enumerate(np.arange(0, np.pi, np.pi / n_direct)):  # gabor方向，0°，45°，90°，135°，共四个
			kern = cv2.getGaborKernel(ksize=(ksize[K], ksize[K]),
									  sigma=1.0,
									  theta=theta,
									  lambd=np.pi / 2,
									  gamma=0.5,
									  psi=0, ktype=cv2.CV_32F)
			# kern /= 1.5 * kern.sum()
			filters[K].append(kern)
		# filters[th_id].append(kern)
	return filters


def make_gabor_conv_weight(filters, channel, cuda=True):
	filter_weights = []
	for idx, ksize_filters in enumerate(filters):
		filter_weights.append([])
		for filter in ksize_filters:
			weight = torch.from_numpy(filter)[None, None, :, :].expand(channel, -1, -1, -1)
			if cuda:
				weight = weight.cuda()
			filter_weights[idx].append(weight)
	return filter_weights


def gabor_batch(weights, batch_input):
	out = []
	for ksize_weights in weights:
		ksize_out = 0
		for w in ksize_weights:
			o = F.conv2d(batch_input, w, padding=w.shape[-1] // 2, stride=1, groups=w.shape[0])
			ksize_out += o
		ksize_out /= len(ksize_weights)
		out.append(ksize_out)
	out = torch.cat(out, dim=1)
	return out


def gen_dense_feat(input_file, out_file, gabor=False):
	NC_OUT = 26
	if gabor:
		NC_OUT = 56
		filters = GaborFilters()
		weights = make_gabor_conv_weight(filters, 10)

	init_data_source = H5DataSource([input_file], BATCH_SIZE, shuffle=False, split=False)
	init_loader = MyDataLoader(init_data_source.h5fids, init_data_source.indices)

	feat_total = None
	label_total = []
	for data, label, _ in tqdm(init_loader):
		data = torch.from_numpy(data).float().cuda()
		data = preprocess_batch(data)

		data = data.transpose(3, 2).transpose(2, 1)  # bs nc w h

		if gabor:
			# TODO: GABOR FEAT
			gabor_data = gabor_batch(weights, data[:, 6:16, :, :])
			data = torch.cat([data, gabor_data], dim=1)

		data = data.view(data.shape[0], data.shape[1], 32 * 32)  # bs, nc, pixes

		mean = data.mean(dim=-1)
		min = data.min(dim=-1)[0]
		max = data.max(dim=-1)[0]
		mid = data.median(dim=-1)[0]
		std = data.std(dim=-1)
		basic_feat = torch.cat([mean, min, max, mid, std], dim=-1).detach().cpu().numpy()
		data = data.detach().cpu().numpy()
		perc = np.percentile(data, [10, 20, 40, 60, 80, 90], axis=-1).transpose((1, 2, 0)).reshape(data.shape[0],
																								   -1)
		batch_feat = np.concatenate([basic_feat, perc], axis=-1)
		if feat_total is None:
			feat_total = batch_feat
		else:
			feat_total = np.concatenate([feat_total, batch_feat], axis=0)

		if label is not None:
			label_total += label.argmax(-1).tolist()

	column = get_dense_column(NC_OUT)

	if len(label_total) == feat_total.shape[0]:
		label_total = np.array(label_total).reshape(-1, 1)
		feat_total = np.concatenate([feat_total, label_total], axis=-1)
		column += ['label']
	print(feat_total.shape)
	dense_df = pd.DataFrame(feat_total, columns=column)
	dense_df.to_csv(out_file, sep=',', index=False)


if __name__ == '__main__':
	train_file = '/home/zydq/Datasets/LCZ/training.h5'
	val_file = '/home/zydq/Datasets/LCZ/validation.h5'
	# testA_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
	testA_file = '/home/zydq/Datasets/LCZ/round2_test_a_20190121.h5'

	testB_file = '/home/zydq/Datasets/LCZ/round2_test_b_20190104.h5'

	dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_train.csv'
	dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_val.csv'
	dense_testA_file = '/home/zydq/Datasets/LCZ/dense_f_test2A.csv'

	dense_testB_file = '/home/zydq/Datasets/LCZ/dense_f_test2B.csv'

	gen_dense_feat(testA_file, dense_testA_file)
