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
from modules.lcz_xception import Xception
from modules.lcz_res_net import resnet10, resnet18, resnet34, resnet50, resnet101
from modules.lcz_senet import se_resnet10_fc512, se_resnet15_fc512
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161

from modules.scheduler import RestartCosineAnnealingLR, CosineAnnealingLR

from modules.losses import FocalCE
from config import  *

SEED = 502
T = 1.5
M = 6
EPOCH = math.ceil(T * M)


model_dir = osp.join(model_root, 'model_' + str(round(time.time() % 100000)))

if not os.path.isdir(model_root):
	os.mkdir(model_root)
if not os.path.isdir(model_dir):
	os.mkdir(model_dir)
cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if __name__ == '__main__':

	mean_std_h5_train = h5py.File(mean_std_train_file, 'r')
	mean_train = torch.from_numpy(np.array(mean_std_h5_train['mean'])).float().cuda()
	std_train = torch.from_numpy(np.array(mean_std_h5_train['std'])).float().cuda()
	mean_std_h5_train.close()

	mean_std_h5_val = h5py.File(mean_std_val_file, 'r')
	mean_val = torch.from_numpy(np.array(mean_std_h5_val['mean'])).float().cuda()
	std_val = torch.from_numpy(np.array(mean_std_h5_val['std'])).float().cuda()
	mean_std_h5_val.close()

	mean = torch.cat([mean_train[np.newaxis, :], mean_val[np.newaxis, :]], dim=0)
	std = torch.cat([std_train[np.newaxis, :], std_val[np.newaxis, :]], dim=0)

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
	elif MODEL == 'DENSE121':
		model = densenet121(N_CHANNEL, 17, drop_rate=0.3)
	elif MODEL == 'DENSE201':
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

	if os.path.isfile(cur_model_path):
		print('load training param, ', cur_model_path)
		state = torch.load(cur_model_path)
		model.load_state_dict(state['cur_model_state'])
		optimizer.load_state_dict(state['cur_opt_state'])
		if 'cur_lr_scheduler_state' in state:
			lr_scheduler.load_state_dict(state['cur_lr_scheduler_state'])
			lr_scheduler.optimizer = optimizer
		epoch_list = range(state['cur_epoch'] + 1, state['cur_epoch'] + 1 + EPOCH)
		global_step = state['cur_step']
	else:
		state = None
		epoch_list = range(EPOCH)
		grade = 1

		global_step = 0

	grade = 0
	print_every = 50
	last_val_step = global_step
	val_every = [1000, 700, 500, 350, 100]
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
				global_step += 1

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

				if global_step - last_val_step == val_every[grade]:
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
							val_input, val_target = prepare_batch(val_data, val_label, f_idx_val, mean_val, std_val)
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
					print('-' * 80)

					if os.path.isfile(cur_model_path):
						state = torch.load(cur_model_path)
					else:
						state = {}

					if (state == {} or state['best_score'] < val_hit / val_sample):
						state['best_model_state'] = model.state_dict()
						state['best_opt_state'] = optimizer.state_dict()
						state['best_lr_scheduler_state'] = lr_scheduler.state_dict()
						state['best_loss'] = val_loss_total / val_step
						state['best_score'] = val_hit / val_sample
						state['best_epoch'] = e
						state['best_step'] = global_step

					state['cur_model_state'] = model.state_dict()
					state['cur_opt_state'] = optimizer.state_dict()
					state['cur_lr_scheduler_state'] = lr_scheduler.state_dict()
					state['cur_epoch'] = e
					state['val_loss'] = val_loss_total / val_step
					state['val_score'] = val_hit / val_sample
					state['cur_step'] = global_step

					torch.save(state, cur_model_path)
