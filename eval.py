import os
import h5py
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from dataloader import MyDataLoader, H5DataSource
from preprocess import prepare_batch
from modules.gac_net import GACNet
from modules.lcz_res_net import resnet18, resnet34, resnet50, resnet101
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161
from sklearn.metrics import classification_report, confusion_matrix
# import torchvision.models as models

BATCH_SIZE = 128
SEED = 502
train_file = '/home/zydq/Datasets/LCZ/training.h5'
val_file = '/home/zydq/Datasets/LCZ/validation.h5'
test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_trainval.h5'
mean_std_file_train = '/home/zydq/Datasets/LCZ/mean_std_f_train.h5'
mean_std_file_val = '/home/zydq/Datasets/LCZ/mean_std_f_val.h5'

MODEL = 'GAC'
# MODEL = 'LCZ'

# model_dir = './checkpoints/model_20930'  # GACNet + class weight  8903  0.770
# model_dir = './checkpoints/model_69737'  # GACNet no class weight  8857  0.780
# model_dir = './checkpoints/model_2234'  # LCZNet + class weight  8773 0.743

# model_dir = './checkpoints/model_1451'  # GACNet + class weight<1 + data_aug
model_dir = './checkpoints/model_11257'  # GACNet + class weight<1 + data_aug + phi
model_dir = './checkpoints/model_85726'  # GACNet + class weight<1 + data_aug + phi

model_dir = './checkpoints/model_80240'  #  FOCAL  trained on train+val
model_dir = './checkpoints/model_18996'  #  FOCAL  indices trained on train+val2

cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if not os.path.isdir('./evaluate/'):
	os.mkdir('./evaluate/')
if __name__ == '__main__':

	mean_std_h5_train = h5py.File(mean_std_file_train, 'r')
	N_CHANNEL = mean_std_h5_train['mean'].shape[-1]
	mean_train = torch.from_numpy(np.array(mean_std_h5_train['mean']).reshape(-1, N_CHANNEL).mean(0)).float().cuda()
	std_train = torch.from_numpy(
		np.sqrt((np.array(mean_std_h5_train['std']).reshape(-1, N_CHANNEL) ** 2).mean(0))).float().cuda()
	# mean = torch.from_numpy(np.array(mean_std_h5['mean'])).float().cuda()
	# std = torch.from_numpy(np.array(mean_std_h5['std'])).float().cuda()
	mean_std_h5_train.close()

	mean_std_h5_val = h5py.File(mean_std_file_val, 'r')
	mean_val = torch.from_numpy(np.array(mean_std_h5_val['mean']).reshape(-1, N_CHANNEL).mean(0)).float().cuda()
	std_val = torch.from_numpy(
		np.sqrt((np.array(mean_std_h5_val['std']).reshape(-1, N_CHANNEL) ** 2).mean(0))).float().cuda()
	# mean = torch.from_numpy(np.array(mean_std_h5['mean'])).float().cuda()
	# std = torch.from_numpy(np.array(mean_std_h5['std'])).float().cuda()
	mean_std_h5_val.close()
	mean = [mean_train, mean_val]
	std = [std_train, std_val]

	mean, std = None, None

	# train val 合并再划分
	# data_source = H5DataSource([train_file, val_file], BATCH_SIZE, split=0.07, seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# 合并再划分 val 中 1:2
	data_source = H5DataSource([train_file, val_file], BATCH_SIZE, [0.02282, 2 / 3], seed=SEED)
	train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)
	class_weights = torch.from_numpy(data_source.class_weights).float().cuda().clamp(0, 1)
	print(class_weights)

	if MODEL == 'GAC':
		group_sizes = [3, 3,
					   3, 3, 2, 2,
					   4, 3, 3]
		model = GACNet(group_sizes, 17, 32)
	elif MODEL == 'RES':
		model = resnet50(N_CHANNEL, 17)
	elif MODEL == 'DENSE':
		model = densenet201(N_CHANNEL, 17, drop_rate=0.3)
	else:
		group_sizes = [3, 3,
					   3, 3, 2, 2,
					   4, 3, 3]
		model = GACNet(group_sizes, 17, 32)

	model = model.cuda()

	best_score = 0

	if os.path.isfile(cur_model_path):
		print('load training param, ', cur_model_path)
		state = torch.load(cur_model_path)
		model.load_state_dict(state['best_model_state'])
		best_score = state['best_score']
	print('best_score:', best_score)

	print('-' * 80)
	print('Evaluating...')
	y_true = []
	y_pred = []
	with torch.no_grad():
		model.eval()
		for val_data, val_label, f_idx in tqdm(val_loader):
			val_input, val_target = prepare_batch(val_data, val_label, f_idx, mean, std)
			# val_out = model(val_input)
			val_out = F.softmax(model(val_input))
			pred = val_out.max(-1)[1].detach().cpu().numpy().tolist()
			gt = val_target.max(-1)[1].detach().cpu().numpy().tolist()

			y_true += gt
			y_pred += pred


	report = classification_report(y_true, y_pred)
	matrix = confusion_matrix(y_true, y_pred)
	np.savetxt('./evaluate/confusion_matrix_' + str(best_score) + '.csv', matrix, delimiter=',', fmt='%d')
	print(report)
	print('-' * 80)
