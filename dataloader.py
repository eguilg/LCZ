import h5py
import torch
import numpy as np


class H5DataSource(object):
	def __init__(self, data_paths, batch_size, split=0.2, shuffle=True, seed=502):
		self.h5fids = []
		for path in data_paths:
			self.h5fids.append(h5py.File(path, 'r'))
		self.batch_nums = [int(fid['sen1'].shape[0] / batch_size) + 1 for fid in self.h5fids]
		self.indices = [(fid,
						 batch_id * batch_size,
						 min((batch_id + 1) * batch_size, self.h5fids[fid]['sen1'].shape[0]))
						for fid in range(len(self.h5fids)) for batch_id in range(self.batch_nums[fid])]
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(self.indices)
		if split is not None:
			split_idx = int(len(self.indices) * split)
			self.val_indices = self.indices[:split_idx]
			self.train_indices = self.indices[split_idx:]


class MyDataLoader(object):
	def __init__(self, h5fids, indices):
		self.h5fids = h5fids
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __iter__(self):
		return _MyDataIter(self)


class _MyDataIter(object):
	def __init__(self, data_loader):
		self.data_loader = data_loader
		self.batch_iter = iter(data_loader.indices)
		self.h5fids = data_loader.h5fids

	def __len__(self):
		return len(self.data_loader.indices)

	def __iter__(self):
		return self

	def __next__(self):
		f_idx, b_start, b_end = next(self.batch_iter)
		y_b = None
		h5fid = self.h5fids[f_idx]
		if 'label' in h5fid.keys():
			y_b = np.array(h5fid['label'][b_start: b_end])
		x_b = np.array(
			np.concatenate(
				(
					h5fid['sen1'][b_start: b_end],
					h5fid['sen2'][b_start: b_end]
				),
				axis=3)
		)
		return x_b, y_b


def prepare_batch(x_b, y_b, mean, std):
	x_b = (x_b - mean[None, :, :, :]) / std[None, :, :, :]
	x_b = torch.from_numpy(x_b).float().cuda()
	if y_b is not None:
		y_b = torch.from_numpy(y_b).max(-1)[1].cuda()
	return x_b, y_b


if __name__ == '__main__':
	train_file = '/home/zydq/Datasets/LCZ/training.h5'
	val_file = '/home/zydq/Datasets/LCZ/validation.h5'

	data_source = H5DataSource([train_file, val_file], 32, split=0.1)
	train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)
	for batch_data, batch_label in train_loader:
		batch_data, batch_label = prepare_batch(batch_data, batch_label)
		print(batch_data.shape, batch_label.shape)
