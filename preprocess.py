import h5py
import numpy as np
from tqdm import tqdm
from dataloader import H5DataSource, MyDataLoader

train_file = '/home/zydq/Datasets/LCZ/training.h5'
val_file = '/home/zydq/Datasets/LCZ/validation.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_val.h5'

if __name__ == '__main__':
	init_data_source = H5DataSource([train_file, val_file], 10000, shuffle=False, split=False)
	init_loader = MyDataLoader(init_data_source.h5fids, init_data_source.indices)
	mean, std, n = 0, 0, 0
	for data, label in tqdm(init_loader):
		mean += np.sum(data, axis=0)
		n += data.shape[0]

	mean /= n

	for data, label in tqdm(init_loader):
		std += np.sum(np.power(data - mean[None, :, :, :], 2), axis=0)

	std = np.sqrt(std / n)

	mean_std_h5 = h5py.File(mean_std_file, 'a')
	mean_std_h5.create_dataset('mean', data=mean)
	mean_std_h5.create_dataset('std', data=std)
	mean_std_h5.close()
