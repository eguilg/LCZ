import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from dataloader import MyDataLoader, H5DataSource
from preprocess import prepare_batch
from modules.gac_net import GACNet
from modules.lcz_res_net import resnet18, resnet34, resnet50
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161

BATCH_SIZE = 32
N_CHANNEL = 26
TEST_B = True

test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
submit_dir = './submit/'
score_dir = './score/'
if TEST_B:
	test_file = '/home/zydq/Datasets/LCZ/round1_test_b_20190104.h5'
	submit_dir = './submit_B/'
	score_dir = './score_B/'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_test.h5'

model_name = 'GAC_mixup0_foc1_weight0_decay0.03'
# model_name = 'GAC_mixup0_foc1_weight0_decay0.03_onval'
model_name = 'GAC_mixup1_foc0_weight0_decay0.02_onval'
model_name = 'GAC_mixup1_foc0_weight0_decay0.02'

model_dir = os.path.join('./checkpoints/', model_name)
MODEL = model_name.split('_')[0]

models = [
	# 'M_curr.ckpt',
	'M_best.ckpt',
	'M_1.ckpt',
	'M_2.ckpt',
	'M_3.ckpt',
	'M_4.ckpt',
	'M_5.ckpt',
	'M_6.ckpt'
]


if not os.path.isdir(submit_dir):
	os.mkdir(submit_dir)
if not os.path.isdir(score_dir):
	os.mkdir(score_dir)
if __name__ == '__main__':

	# mean_std_h5 = h5py.File(mean_std_file, 'r')
	# # N_CHANNEL = mean_std_h5['mean'].shape[-1]
	# mean = [torch.from_numpy(np.array(mean_std_h5['mean']).reshape(-1, N_CHANNEL).mean(0)).cuda()]
	# std = [torch.from_numpy(np.sqrt((np.array(mean_std_h5['std']).reshape(-1, N_CHANNEL) ** 2).mean(0))).cuda()]
	# # mean = torch.from_numpy(np.array(mean_std_h5['mean'])).float().cuda()
	# # std = torch.from_numpy(np.array(mean_std_h5['std'])).float().cuda()
	# mean_std_h5.close()

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

	data_source = H5DataSource([test_file], BATCH_SIZE, shuffle=False)
	test_loader = MyDataLoader(data_source.h5fids, data_source.indices)

	n_model = 0
	ensembled_pred = None
	ensembled_score = 0
	for ckpt_name in models:
		ckpt_path = os.path.join(model_dir, ckpt_name)
		mean, std = None, None
		if os.path.isfile(ckpt_path):
			print('load training param, ', ckpt_path)
			state = torch.load(ckpt_path)
			model.load_state_dict(state['model_state'])
			m_score = state['score']
			m_loss = state['loss']
			print('score:', m_score)
			print('loss:', m_loss)
			print('-' * 80)
			print('Testing...')

			total_score = None
			with torch.no_grad():
				model.eval()
				for test_data, _, fidx in tqdm(test_loader):
					time.sleep(0.02)
					test_input, _ = prepare_batch(test_data, None, fidx, mean, std)
					test_out = F.softmax(model(test_input), -1)
					score = test_out.detach().cpu().numpy()
					if total_score is None:
						# total_pred = pred
						total_score = score
					else:
						# total_pred = np.concatenate([total_pred, pred])
						total_score = np.concatenate([total_score, score])
			ensembled_score += total_score
			n_model += 1
			del state

	ensembled_score /= n_model
	ensembled_pred = ensembled_score.argmax(-1)
	submit = np.eye(17)[ensembled_pred.reshape(-1)]

	np.savetxt(os.path.join(submit_dir, model_name + '.csv'), submit, delimiter=',', fmt='%d')
	np.savetxt(os.path.join(score_dir, model_name + '.csv'), ensembled_score, delimiter=',', fmt='%.5f')
	print('completed!')
	print('-' * 80)
