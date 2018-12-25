import os
import time
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from dataloader import MyDataLoader, H5DataSource
from preprocess import prepare_batch
from modules.lcz_net import LCZNet
from modules.gac_net import GACNet
from modules.lcz_res_net import resnet18, resnet34, resnet50, resnet101
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161

from modules.losses import FocalCE, SoftCE

SEED = 502
EPOCH = 150
BATCH_SIZE = 64
LR = 1.25e-4
LAMBDA = 4
USE_CLASS_WEIGHT = True
MIX_UP = False
FOCAL = False
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

model_dir = './checkpoints/model_' + str(round(time.time() % 100000))

# model_dir = './checkpoints/model_80240'


# model_dir = './checkpoints/model_20930'  #GACNet with class weight  8903
# model_dir = './checkpoints/model_69737'  #GACNet with no class weight  8857

# model_dir = './checkpoints/model_85726_ft'  # GACNet + class weight<1 + data_aug + phi
if not os.path.isdir('./checkpoints/'):
	os.mkdir('./checkpoints/')
if not os.path.isdir(model_dir):
	os.mkdir(model_dir)
cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
	data_source = H5DataSource([train_file]+[val_file]*10, BATCH_SIZE, [0.114] + [0]*10, seed=SEED)
	train_loader = MyDataLoader(data_source.h5fids, data_source.train_indices)
	val_loader = MyDataLoader(data_source.h5fids, data_source.val_indices)

	class_weights = torch.from_numpy(data_source.class_weights).float().cuda()
	node_class_weights = torch.from_numpy(data_source.node_class_weights).float().cuda()
	#  origin class weights
	class_weights = (1/17 / class_weights).clamp(0, 1)
	node_class_weights = (1 / 5 / node_class_weights).clamp(0, 1)
	# focal class weights
	# class_weights = (1 - class_weights) ** LAMBDA
	# node_class_weights = (1 - node_class_weights) ** LAMBDA


	print(node_class_weights)
	print(class_weights)

	if MODEL == 'LCZ':
		model = LCZNet(channel=N_CHANNEL, n_class=17, base=64, dropout=0.3)
	elif MODEL == 'GAC':
		group_sizes = [3, 3,
					   3, 3, 2, 2,
					   4, 3, 3]
		class_nodes = [3, 3, 4, 4, 3]
		model = GACNet(group_sizes, class_nodes, 32)
	elif MODEL == 'RES':
		class_nodes = [3, 3, 4, 4, 3]
		# model = resnet34(N_CHANNEL, class_nodes)
		model = resnet18(N_CHANNEL, class_nodes)
	elif MODEL == 'DENSE':
		class_nodes = [3, 3, 4, 4, 3]
		model = densenet201(N_CHANNEL, class_nodes, drop_rate=0.3)
	else:
		model = LCZNet(channel=N_CHANNEL, n_class=17, base=64, dropout=0.3)

	model = model.cuda()
	model_param_num = 0
	for param in list(model.parameters()):
		model_param_num += param.nelement()
	print('num_params: %d' % (model_param_num))
	if FOCAL or MIX_UP:
		crit = SoftCE
		if FOCAL:
			crit = FocalCE
	else:
		crit = nn.NLLLoss

	if USE_CLASS_WEIGHT:
		criteria1 = crit(weight=class_weights).cuda()
		criteria2 = crit(weight=node_class_weights).cuda()
	else:
		criteria1 = crit().cuda()
		criteria2 = crit().cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
	lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=1e-7)

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
				train_input, train_target = prepare_batch(train_data, train_label, f_idx_train, mean, std, aug=True, mix=MIX_UP)
				# print(train_input.shape)
				model.train()
				optimizer.zero_grad()

				train_node_out, train_out = model(train_input)
				if MIX_UP:
					y_1, y_2, lam = train_target
					loss = mixup_criterion(criteria1, train_out, y_1[:, 5:], y_2[:, 5:], lam)
					loss_node = mixup_criterion(criteria2, train_node_out, y_1[:, :5], y_2[:, :5], lam)
					train_target = lam * y_1 + (1 - lam) * y_2
				elif FOCAL:
					loss = criteria1(train_out, train_target[:, 5:])
					loss_node = criteria2(train_node_out, train_target[:, :5])
				else:
					loss = criteria1(torch.log(train_out), train_target[:, 5:].max(-1)[1])
					loss_node = criteria2(torch.log(train_node_out), train_target[:, :5].max(-1)[1])

				train_hit += (train_out.max(-1)[1] == train_target[:, 5:].max(-1)[1]).sum().item()


				# loss.backward()
				loss_total = loss + loss_node
				loss_total.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

				lr_scheduler.step()
				optimizer.step()

				loss_print += loss.item()
				step += 1
				global_step += 1

				# train_hit += (train_out.max(-1)[1] == train_target).sum().item()

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
							val_input, val_target = prepare_batch(val_data, val_label, f_idx_val, mean, std)
							# val_out = model(val_input)
							val_node_out, val_out = model(val_input)
							if FOCAL or MIX_UP:
								val_loss_total += criteria1(val_out, val_target[:, 5:]).item()
							else:
								val_loss_total += criteria1(torch.log(val_out), val_target[:, 5:].max(-1)[1]).item()
							val_hit += (val_out.max(-1)[1] == val_target[:, 5:].max(-1)[1]).sum().item()
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

					if state == {} or state['best_score'] < val_hit / val_sample:
						state['best_model_state'] = model.state_dict()
						state['best_opt_state'] = optimizer.state_dict()
						state['best_lr_scheduler_state'] = lr_scheduler.state_dict()
						state['best_loss'] = val_loss_total / val_step
						state['best_score'] = val_hit / val_sample
						state['best_epoch'] = e
						state['best_step'] = global_step


						# if state['best_score'] > 0.90:
						# 	grade = 3
						# elif state['best_score'] > 0.89:
						# 	grade = 2
						# elif state['best_score'] > 0.88:
						# 	grade = 1
						# else:
						# 	grade = 0
					# if state == {} or state['val_score'] < val_hit / val_sample:
					# 	val_no_improve = 0
					# else:
					# 	val_no_improve += 1
					#
					# if val_no_improve >= int(drop_lr_frq):
					# 	print('dropping lr...')
					# 	val_no_improve = 0
					# 	drop_lr_frq += 1
					# 	lr_total = 0
					# 	lr_num = 0
					# 	for param_group in optimizer.param_groups:
					# 		if param_group['lr'] > 1e-5:
					# 			param_group['lr'] *= 0.6
					# 		else:
					# 			param_group['lr'] = LR
					#
					# 		lr_total += param_group['lr']
					# 		lr_num += 1
					# 	drop_lr_frq = 1
					# 	print('curr avg lr is {}'.format(lr_total / lr_num))

					state['cur_model_state'] = model.state_dict()
					state['cur_opt_state'] = optimizer.state_dict()
					state['cur_lr_scheduler_state'] = lr_scheduler.state_dict()
					state['cur_epoch'] = e
					state['val_loss'] = val_loss_total / val_step
					state['val_score'] = val_hit / val_sample
					state['cur_step'] = global_step

					torch.save(state, cur_model_path)
