import torch  # for speed
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataloader import H5DataSource, MyDataLoader
from preprocess import preprocess_batch

if __name__ == '__main__':
	train_file = '/home/zydq/Datasets/LCZ/training.h5'
	val_file = '/home/zydq/Datasets/LCZ/validation.h5'
	test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
	mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_train.h5'
	raw_file = [train_file, val_file, test_file]

	dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_train.csv'
	dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_val.csv'
	dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_test.csv'
	dist = [dense_train_file, dense_val_file, dense_test_file]
	column = ['mean_' + str(i) for i in range(26)] + \
			 ['min_' + str(i) for i in range(26)] + \
			 ['max_' + str(i) for i in range(26)] + \
			 ['mid_' + str(i) for i in range(26)] + \
			 ['std_' + str(i) for i in range(26)] + \
			 ['per' + str(j) + '_' + str(i) for i in range(26) for j in [5, 20, 40, 60, 80, 95]]
	for fid in range(3):
		init_data_source = H5DataSource([raw_file[fid]], 3000, shuffle=False, split=False)
		init_loader = MyDataLoader(init_data_source.h5fids, init_data_source.indices)

		feat_total = None
		label_total = []
		for data, label, _ in tqdm(init_loader):
			data = torch.from_numpy(data).float().cuda()
			data = preprocess_batch(data)
			data = data.view(data.shape[0], 32 * 32, data.shape[-1]).transpose(2, 1)  # bs, nc, pixes

			mean = data.mean(dim=-1)
			min = data.min(dim=-1)[0]
			max = data.max(dim=-1)[0]
			mid = data.median(dim=-1)[0]
			std = data.std(dim=-1)
			basic_feat = torch.cat([mean, min, max, mid, std], dim=-1).detach().cpu().numpy()
			data = data.detach().cpu().numpy()
			perc = np.percentile(data, [10, 20, 40, 60, 80, 90], axis=-1).transpose((1, 2, 0)).reshape(data.shape[0],-1)
			batch_feat = np.concatenate([basic_feat, perc], axis=-1)
			if feat_total is None:
				feat_total = batch_feat
			else:
				feat_total = np.concatenate([feat_total, batch_feat], axis=0)

			if label is not None:
				label_total += label.argmax(-1).tolist()

		column_names = column.copy()
		if len(label_total) == feat_total.shape[0]:
			label_total = np.array(label_total).reshape(-1,1)
			feat_total = np.concatenate([feat_total, label_total], axis=-1)
			column_names += ['label']
		print(feat_total.shape)
		dense_df = pd.DataFrame(feat_total, columns=column_names)
		dense_df.to_csv(dist[fid], sep=',',index=False)





