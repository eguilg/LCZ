import torch  # for speed
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
# from gen_dense_feature import GaborFilters, make_gabor_conv_weight, gabor_batch

train_path = '/home/zydq/Datasets/LCZ/training.h5'
vali_path = '/home/zydq/Datasets/LCZ/validation.h5'

train_h5 = h5py.File(train_path, 'r')
vali_h5 = h5py.File(vali_path, 'r')
label_names = ['LCZ 1:  Compact high-rise', 'LCZ 2:  Compact mid-rise', 'LCZ 3:  Compact low-rise',
			   'LCZ 4:  Open high-rise', 'LCZ 5:  Open mid-rise', 'LCZ 6:  Open low-rise',
			   'LCZ 7:  Lightweight low-rise', 'LCZ 8:  Large low-rise', 'LCZ 9:  Sparsely built',
			   'LCZ 10:  Heavy industry', 'LCZ A:  Dense trees', 'LCZ B:  Scattered trees',
			   'LCZ C:  Bush, scrub', 'LCZ D:  Low plants', 'LCZ E:  Bare rock or paved',
			   'LCZ F:  Bare soil or sand', 'LCZ G:  Water']

def GaborFilters(ksize=None, n_direct=4):
	filters = []
	if ksize is None:
		ksize = [3, 5, 7]
	for th_id, theta in enumerate(np.arange(0, np.pi, np.pi / n_direct)):  # gabor方向，0°，45°，90°，135°，共四个
		filters.append([])
		for K in range(len(ksize)):
			kern = cv2.getGaborKernel(ksize=(ksize[K], ksize[K]),
									  sigma=1,
									  theta=theta,
									  lambd=5,
									  gamma=0.1,
									  psi=0, ktype=cv2.CV_32F)
			kern /= 1.5 * kern.sum()
			# filters[K].append(kern)
			filters[th_id].append(kern)
	return filters

def gabor_img(filters, img):
	out = []
	# img = img.transpose(2,0,1)
	print(img.shape)
	for ksize_fs in filters:
		ksize_out = 0
		for f in ksize_fs:
			o = cv2.filter2D(np.uint8(img*255), cv2.CV_8UC3, f).astype(float)/255
			ksize_out += o
		ksize_out /= len(ksize_fs)
		out.append(ksize_out[None,:,:,:])
	out = np.concatenate(out, axis=0)
	return out

filters = GaborFilters(ksize=[5])


def view_gabor_data(datasource, num, dims=None):
	if dims is None:
		dims = [2, 1, 0]
	img = np.array(datasource['sen2'][num, :, :, dims])

	import matplotlib.pyplot as plt
	out = gabor_img(filters, img)

	RGB = np.concatenate([datasource['sen2'][num, :, :, [2]],
						  datasource['sen2'][num, :, :, [1]],
						  datasource['sen2'][num, :, :, [0]]], -1)

	plt.figure(figsize=(10, 5))

	plt.subplot(251)
	plt.imshow(out[0]*3)
	plt.subplot(252)
	plt.imshow(out[1]*3)
	plt.subplot(253)
	plt.imshow(out[2]*3)
	plt.subplot(254)
	plt.imshow(out[3]*3)
	plt.subplot(255)
	plt.imshow(RGB*3)

	plt.show()
	# for fs in filters:
	# 	for f in fs:
	# 		plt.subplot()
	# 		plt.imshow(f)
	# 		plt.show()
	print(label_names[np.argmax(datasource['label'][num])])


if __name__ == '__main__':
	view_gabor_data(train_h5, 83, [0,1,2])
