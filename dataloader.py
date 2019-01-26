import h5py
import numpy as np


class SampledDataSorce(object):
	def __init__(self, data_paths, batch_size, sample_rate=None, split=0.1, shuffle=True, seed=502):
		self.h5fids = []
		self.bs = []
		self.batch_nums = []
		if sample_rate is None:
			sample_rate = [1 / len(data_paths)] * len(data_paths)
		for idx, path in enumerate(data_paths):
			self.h5fids.append(h5py.File(path, 'r'))
			self.bs.append(int(sample_rate[idx] * batch_size))
			self.batch_nums.append(int(self.h5fids[-1]['sen1'].shape[0] / self.bs[-1]) + 1)

		self.h5file_indices = [[(fid,
								 batch_id * self.bs[fid],
								 min((batch_id + 1) * self.bs[fid], self.h5fids[fid]['sen1'].shape[0]))
								for batch_id in range(self.batch_nums[fid])] for fid in range(len(self.h5fids))]

		if shuffle:
			for h5_indice in self.h5file_indices:
				np.random.seed(seed)
				np.random.shuffle(h5_indice)
		self.val_indices = []
		self.train_indices = []

		self.indices = [
			(self.h5file_indices[fid][bid % self.batch_nums[fid]] for fid in range(len(self.h5file_indices))) for
			bid in range(max(self.batch_nums))]

		if 0 < split < 1:
			split_idx = min([int(batch_num * split) for batch_num in self.batch_nums])
			self.val_indices = [[h5file_indice[-bid] for h5file_indice in self.h5file_indices] for bid in
								range(1, split_idx + 1)]
			self.train_indices = [
				[self.h5file_indices[fid][bid % (self.batch_nums[fid] - split_idx)] for fid in
				 range(len(self.h5file_indices))] for
				bid in range(max(self.batch_nums) - split_idx)]

			print(self.train_indices)
			print(self.val_indices)

		if 'label' in self.h5fids[0].keys():
			self.class_weights = np.concatenate([h5['label'] for h5 in self.h5fids], axis=0).mean(axis=0)
			self.node_class_weights = np.array([self.class_weights[:3].sum(),
												self.class_weights[3:6].sum(),
												self.class_weights[6:10].sum(),
												self.class_weights[10:14].sum(),
												self.class_weights[14:17].sum()])
			print(self.node_class_weights)
			print(self.class_weights)


class H5DataSource(object):
	def __init__(self, data_paths, batch_size, val_ratios=None, split=0.1, shuffle=True, seed=502):
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

		if val_ratios is not None:
			split_idxs = [int(self.batch_nums[id] * val_ratios[id]) for id in range(len(val_ratios))]
			print(split_idxs)
			fid_indices = [list(filter(lambda f: f[0] == fid, self.indices)) for fid in range(len(self.h5fids))]
			self.val_indices = []
			self.train_indices = []
			for split_idx, indices in zip(split_idxs, fid_indices):
				self.val_indices += indices[:split_idx]
				self.train_indices += indices[split_idx:]

			if shuffle:
				np.random.seed(seed)
				np.random.shuffle(self.train_indices)
				np.random.seed(seed)
				np.random.shuffle(self.val_indices)

			print(self.train_indices)
			print(self.val_indices)

		elif split is not None:
			split_idx = int(len(self.indices) * split)
			self.val_indices = self.indices[:split_idx]
			self.train_indices = self.indices[split_idx:]
		if 'label' in self.h5fids[0].keys():
			self.class_weights = np.concatenate([h5['label'] for h5 in self.h5fids], axis=0).mean(axis=0)
			self.node_class_weights = np.array([self.class_weights[:3].sum(),
												self.class_weights[3:6].sum(),
												self.class_weights[6:10].sum(),
												self.class_weights[10:14].sum(),
												self.class_weights[14:17].sum()])
			print(self.node_class_weights)
			print(self.class_weights)


class MyDataLoader(object):
	def __init__(self, h5fids, indices):
		self.h5fids = h5fids
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __iter__(self):
		return _MyDataIter(self)

	def shuffle_batch(self, seed=502):
		np.random.seed(seed)
		np.random.shuffle(self.indices)


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
		# f_idx, b_start, b_end = next(self.batch_iter)
		next_batch = next(self.batch_iter)
		if isinstance(next_batch, tuple):
			next_batch = [next_batch]

		x_b, y_b, f_idx_b = [], [], []
		for (f_idx, b_start, b_end) in next_batch:

			h5fid = self.h5fids[f_idx]
			if 'label' in h5fid.keys():
				y_b.append(np.array(h5fid['label'][b_start: b_end]))
			x_b.append(np.array(
				np.concatenate(
					(
						h5fid['sen1'][b_start: b_end],
						h5fid['sen2'][b_start: b_end]
					),
					axis=3)
			))
			f_idx_b.append([f_idx] * (b_end - b_start))
		x_b = np.concatenate(x_b, axis=0)
		f_idx_b = np.concatenate(f_idx_b, axis=0)
		if len(y_b) != 0:
			y_b = np.concatenate(y_b, axis=0)
		else:
			y_b = None
		return x_b, y_b, f_idx_b


if __name__ == '__main__':
	from config import *

	data_source = H5DataSource([train_file, val_file], 256, [0.02282, 2 / 3], split=0.05, seed=502)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	sampled_source = SampledDataSorce([train_file, val_file], 64, [0.3, 0.7])
# for batch_data, batch_label in train_loader:
# 	# batch_data, batch_label = prepare_batch(batch_data, batch_label)
# 	print(batch_data.shape, batch_label.shape)
# print(data_source.val_indices)
