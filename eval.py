import os
import h5py
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from dataloader import MyDataLoader, H5DataSource, SampledDataSorce
from preprocess import prepare_batch
from modules.gac_net import GACNet
from modules.resnext import resnext_ys
from modules.lcz_res_net import resnet10, resnet18, resnet34, resnet50
from modules.lcz_senet import se_resnet_ys, se_resnet10_fc512, se_resnet15_fc512
from modules.lcz_xception import Xception
from modules.lcz_dense_net import densenet_ys, densenet121, densenet169, densenet201, densenet161
from sklearn.metrics import classification_report, confusion_matrix
# import torchvision.models as models
from config import *


model_dir = osp.join(model_root, model_name)
# model_dir = './checkpoints/RES10_mixup0_foc0_weight0_decay0.01_draft'

cur_model_path = os.path.join(model_dir, 'M_best.ckpt')
# cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if not os.path.isdir('./evaluate/'):
	os.mkdir('./evaluate/')
if __name__ == '__main__':

	mean, std = None, None
	if ZSCORE:
		mean_std_h5_val = h5py.File(mean_std_val_file, 'r')
		mean = torch.from_numpy(np.array(mean_std_h5_val['mean'])).float().cuda()
		std = torch.from_numpy(np.array(mean_std_h5_val['std'])).float().cuda()
		mean_std_h5_val.close()



	# train val 合并再划分
	# data_source = H5DataSource([train_file, val_file], BATCH_SIZE, split=0.07, seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# 合并再划分 val 中 1:2
	# data_source = H5DataSource([train_file, val_file], BATCH_SIZE, [0.02282, 2 / 3], seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# train val 固定比例 1 : 1
	data_source = SampledDataSorce([train_file, val_file], BATCH_SIZE, sample_rate=[0.5, 0.5], seed=SEED)
	train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# train val 固定比例 1:7
	# data_source = SampledDataSorce([train_file, val_file], BATCH_SIZE, sample_rate=[0.125, 0.875], seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	class_weights = torch.from_numpy(data_source.class_weights).float().cuda().clamp(0, 1)
	print(class_weights)

	if MODEL == 'GAC':
		group_sizes = [3, 3,
					   3, 3, 2, 2,
					   4, 3, 3]
		model = GACNet(group_sizes, 17, 32)
	elif MODEL == 'XCEPTION':
		model = Xception(N_CHANNEL, 17)
	elif MODEL == 'RES10':
		model = resnet10(N_CHANNEL, 17)
	elif MODEL == 'RES18':
		model = resnet18(N_CHANNEL, 17)
	elif MODEL == 'SE-RES10':
		model = se_resnet10_fc512(N_CHANNEL, 17)
	elif MODEL == 'SE-RES15':
		model = se_resnet15_fc512(N_CHANNEL, 17)
	elif MODEL == 'SE-RES-YS':
		model = se_resnet_ys(N_CHANNEL, 17)
	elif MODEL == 'RESNEXT':
		model = resnext_ys(N_CHANNEL, 17)
	elif MODEL == 'DENSE121':
		model = densenet121(N_CHANNEL, 17, drop_rate=0.3)
	elif MODEL == 'DENSE201':
		model = densenet201(N_CHANNEL, 17, drop_rate=0.3)
	elif MODEL == 'DENSE-YS':
		model = densenet_ys(N_CHANNEL, num_classes=17)
	else:
		group_sizes = [3, 3,
					   3, 3, 2, 2,
					   4, 3, 3]
		model = GACNet(group_sizes, 17, 32)
	model = model.cuda()

	best_score = 0
	best_loss = np.inf
	if os.path.isfile(cur_model_path):
		print('load training param, ', cur_model_path)
		state = torch.load(cur_model_path)
		model.load_state_dict(state['model_state'])
		best_score = state['score']
		best_loss = state['loss']
	print('best_score:', best_score)
	print('best_loss:', best_loss)

	print('-' * 80)
	print('Evaluating...')
	y_true = []
	y_pred = []
	with torch.no_grad():
		model.eval()
		for val_data, val_label, f_idx in tqdm(val_loader):
			val_input, val_target = prepare_batch(val_data, val_label, f_idx, mean, std)
			# val_out = model(val_input)
			val_out = F.softmax(model(val_input), dim=-1)
			pred = val_out.max(-1)[1].detach().cpu().numpy().tolist()
			gt = val_target.max(-1)[1].detach().cpu().numpy().tolist()

			y_true += gt
			y_pred += pred


	report = classification_report(y_true, y_pred)
	matrix = confusion_matrix(y_true, y_pred)
	np.savetxt('./evaluate/confusion_matrix_' + str(best_score) + '.csv', matrix, delimiter=',', fmt='%d')
	print(report)
	print('-' * 80)
