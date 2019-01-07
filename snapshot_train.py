import os
import time
import math
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from dataloader import MyDataLoader, H5DataSource, SampledDataSorce
from preprocess import prepare_batch, mixup_data, mixup_criterion
from modules.gac_net import GACNet
from modules.lcz_res_net import resnet18, resnet34, resnet50, resnet101
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161

from modules.scheduler import RestartCosineAnnealingLR, CosineAnnealingLR

from modules.losses import FocalCE

SEED = 502
T = 1.5
M = 6
EPOCH = int(math.ceil(T * M))
BATCH_SIZE = 64
LR = 1e-4
DECAY = 3e-2
USE_CLASS_WEIGHT = False
MIX_UP = False
FOCAL = True
MIX_UP_ALPHA = 1.0
N_CHANNEL = 26

MODEL = 'GAC'
# MODEL = 'DENSE'
# MODEL = 'RES'
# MODEL = 'LCZ'


train_file = '/home/zydq/Datasets/LCZ/training.h5'
val_file = '/home/zydq/Datasets/LCZ/validation.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_trainval.h5'
mean_std_file_train = '/home/zydq/Datasets/LCZ/mean_std_f_train.h5'
mean_std_file_val = '/home/zydq/Datasets/LCZ/mean_std_f_val.h5'

name_arg = [MODEL, 'mixup' + str(int(MIX_UP)), 'foc' + str(int(FOCAL)), 'weight' + str(int(USE_CLASS_WEIGHT)),
			'decay' + str(DECAY)]
model_name = '_'.join(name_arg)
model_dir = './checkpoints/' + model_name

if not os.path.isdir('./checkpoints/'):
	os.mkdir('./checkpoints/')
if not os.path.isdir(model_dir):
	os.mkdir(model_dir)

cur_model_path = os.path.join(model_dir, 'M_curr.ckpt')
best_model_path = os.path.join(model_dir, 'M_best.ckpt')

if __name__ == '__main__':

	mean_std_h5_train = h5py.File(mean_std_file_train, 'r')
	# N_CHANNEL = mean_std_h5_train['mean'].shape[-1]
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

	# 官方train val
	# train_source = H5DataSource([train_file], BATCH_SIZE, split=None, seed=SEED)
	# val_source = H5DataSource([val_file], BATCH_SIZE, shuffle=False, split=None)
	# train_loader = MyDataLoader(train_source.h5fids, train_source.indices)
	# val_loader = MyDataLoader(val_source.h5fids, val_source.indices)

	# 只用val
	# data_source = H5DataSource([val_file], BATCH_SIZE, split=0.1, seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# 合并再划分 val 中 1:2
	# data_source = H5DataSource([train_file, val_file], BATCH_SIZE, [0.02282, 2 / 3], seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# 合并再划分 val*10 全在训练及
	# data_source = H5DataSource([train_file]+[val_file]*10, BATCH_SIZE, [0.114] + [0]*10, seed=SEED)
	# train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	# val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	# train val 固定比例
	data_source = SampledDataSorce([train_file, val_file], BATCH_SIZE, sample_rate=[0.5, 0.5], seed=SEED)
	train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	class_weights = torch.from_numpy(data_source.class_weights).float().cuda()
	node_class_weights = torch.from_numpy(data_source.node_class_weights).float().cuda()
	#  origin class weights
	class_weights = (1 / 17 / class_weights).clamp(0, 1)
	node_class_weights = (1 / 5 / node_class_weights).clamp(0, 1)

	print(node_class_weights)
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
	model_param_num = 0
	for param in list(model.parameters()):
		model_param_num += param.nelement()
	print('num_params: %d' % (model_param_num))
	if FOCAL:
		crit = FocalCE
	else:
		crit = nn.CrossEntropyLoss

	if USE_CLASS_WEIGHT:
		criteria = crit(weight=class_weights).cuda()
	else:
		criteria = crit().cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
	lr_scheduler = RestartCosineAnnealingLR(optimizer, T_max=int(T * len(train_loader)), eta_min=1e-7)

	if os.path.isfile(best_model_path):
		print('load training param, ', best_model_path)
		best_state = torch.load(best_model_path)
		best_acc = best_state['score']
		del best_state
	else:
		best_acc = 0

	if os.path.isfile(cur_model_path):
		print('load training param, ', cur_model_path)
		state = torch.load(cur_model_path)
		model.load_state_dict(state['model_state'])
		optimizer.load_state_dict(state['opt_state'])
		if 'lr_scheduler_state' in state:
			lr_scheduler.load_state_dict(state['scheduler_state'])
			lr_scheduler.optimizer = optimizer
			global_step = lr_scheduler.last_epoch + 1
		else:
			global_step = 0
	else:
		state = None
		epoch_list = range(EPOCH)
		grade = 1
		global_step = 0

	grade = 0
	print_every = 50
	last_val_step = global_step
	val_every = 1000
	drop_lr_frq = 1
	val_no_improve = 0
	loss_print = 0
	train_hit = 0
	train_sample = 0

	for e in epoch_list:
		step = 0
		train_loader.shuffle_batch(SEED)
		with tqdm(total=len(train_loader)) as bar:
			for i, (train_data, train_label, f_idx_train) in enumerate(train_loader):
				train_input, train_target = prepare_batch(train_data, train_label, f_idx_train, mean, std, aug=True)

				model.train()
				optimizer.zero_grad()

				if MIX_UP:
					train_input, y_1, y_2, lam = mixup_data(train_input, train_target, alpha=MIX_UP_ALPHA)
					train_out = model(train_input)
					loss = mixup_criterion(criteria, train_out, y_1.max(-1)[1], y_2.max(-1)[1], lam)
					train_target = lam * y_1 + (1 - lam) * y_2
				elif FOCAL:
					train_out = model(train_input)
					loss = criteria(train_out, train_target)
				else:
					train_out = model(train_input)
					loss = criteria(train_out, train_target.max(-1)[1])

				train_hit += (train_out.max(-1)[1] == train_target.max(-1)[1]).sum().item()

				loss.backward()

				torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

				lr_scheduler.step()
				optimizer.step()

				loss_print += loss.item()
				step += 1
				global_step = lr_scheduler.last_epoch + 1

				train_sample += train_target.size()[0]

				if global_step % print_every == 0:
					bar.update(min(print_every, step))
					time.sleep(0.02)
					print('Epoch: [{}][{}/{}]\t'
						  'loss: {:.4f}\t'
						  'acc {:.4f}'
						  .format(e, step, len(train_loader),
								  loss_print / print_every,
								  train_hit / train_sample
								  ))

					loss_print = 0
					train_hit = 0
					train_sample = 0

				if global_step - last_val_step == val_every or global_step % int(T * lr_scheduler.T_max) == 0:
					print('-' * 80)
					print('Evaluating...')
					last_val_step = global_step
					val_loss_total = 0
					val_step = 0
					val_hit = 0
					val_sample = 0
					with torch.no_grad():
						model.eval()
						for val_data, val_label, f_idx_val in val_loader:
							val_input, val_target = prepare_batch(val_data, val_label, f_idx_val, mean, std)
							val_out = model(val_input)
							if FOCAL:
								val_loss_total += criteria(train_out, train_target).item()
							else:
								val_loss_total += criteria(val_out, val_target.max(-1)[1]).item()

							val_hit += (val_out.max(-1)[1] == val_target.max(-1)[1]).sum().item()
							val_sample += val_target.size()[0]
							val_step += 1

					print('Val Epoch: [{}][{}/{}]\t'
						  'loss: {:.4f}\t'
						  'acc: {:.4f}\t'
						  .format(e, step, len(train_loader),
								  val_loss_total / val_step,
								  val_hit / val_sample))

					if os.path.isfile(cur_model_path):
						state = torch.load(cur_model_path)
					else:
						state = {}

					state['model_state'] = model.state_dict()
					state['opt_state'] = optimizer.state_dict()
					state['lr_scheduler_state'] = lr_scheduler.state_dict()
					state['loss'] = val_loss_total / val_step
					state['score'] = val_hit / val_sample

					torch.save(state, cur_model_path)
					print('saved curr model to ', cur_model_path)

					if global_step % int(T * lr_scheduler.T_max) == 0:
						M_name = '_'.join(['M', str(global_step // int(T * lr_scheduler.T_max))] + '.ckpt')
						M_path = os.path.join(model_dir, M_name)

						torch.save(state, M_path)
						print('saved snapshot model to ', M_path)
					elif best_acc <= state['score']:
						torch.save(state, best_model_path)
						print('saved best model to ', best_model_path)

					if best_acc <= state['score']:
						best_acc = state['score']

					print('curr best acc:', best_acc)
					print('-' * 80)
