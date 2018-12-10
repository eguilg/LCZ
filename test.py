import os
import h5py
from tqdm import tqdm
import torch
import numpy as np
from dataloader import MyDataLoader, H5DataSource
from preprocess import prepare_batch
from modules.lcz_net import LCZNet
from modules.gac_net import GACNet

# import torchvision.models as models

BATCH_SIZE = 32
test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f39_trainval.h5'

MODEL = 'GAC'
# MODEL = 'LCZ'

# model_dir = './checkpoints/model_20930'  # GACNet + class weight  8903  770
# model_dir = './checkpoints/model_69737'  # GACNet no class weight  8857  780
# model_dir = './checkpoints/model_2234'  # LCZNet + class weight  8773 743

model_dir = './checkpoints/model_29624'  # GACNet + class weight<1

cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if not os.path.isdir('./submit/'):
	os.mkdir('./submit/')
if not os.path.isdir('./score/'):
	os.mkdir('./score/')
if __name__ == '__main__':

	mean_std_h5 = h5py.File(mean_std_file, 'r')
	N_CHANNEL = mean_std_h5['mean'].shape[-1]
	mean = torch.from_numpy(np.array(mean_std_h5['mean']).reshape(-1, N_CHANNEL).mean(0)).cuda()
	std = torch.from_numpy(np.sqrt((np.array(mean_std_h5['std']).reshape(-1, N_CHANNEL) ** 2).mean(0))).cuda()
	# mean = torch.from_numpy(np.array(mean_std_h5['mean'])).float().cuda()
	# std = torch.from_numpy(np.array(mean_std_h5['std'])).float().cuda()
	mean_std_h5.close()

	data_source = H5DataSource([test_file], BATCH_SIZE, shuffle=False)
	test_loader = MyDataLoader(data_source.h5fids, data_source.indices)

	if MODEL == 'LCZ':
		model = LCZNet(channel=N_CHANNEL, n_class=17, base=64, dropout=0.3)
	elif MODEL == 'GAC':
		# group_sizes = [2, 2, 2, 2, 2,
		# 			   3, 3, 1, 1, 2]
		group_sizes = [3, 3, 3, 3, 3, 3, 3, 4, 4,
					   3, 3, 1, 1, 2]
		class_nodes = [3, 3, 4, 4, 3]
		model = GACNet(group_sizes, class_nodes, 32)
	else:
		model = LCZNet(channel=N_CHANNEL, n_class=17, base=64, dropout=0.3)

	model = model.cuda()

	best_score = 0

	if os.path.isfile(cur_model_path):
		print('load training param, ', cur_model_path)
		state = torch.load(cur_model_path)
		model.load_state_dict(state['best_model_state'])
		best_score = state['best_score']
	print('best_score:', best_score)

	print('-' * 80)
	print('Testing...')
	total_pred = None
	total_score = None
	with torch.no_grad():
		model.eval()
		for test_data, _ in tqdm(test_loader):
			test_input, _ = prepare_batch(test_data, None, mean, std)
			test_node_out, test_out = model(test_input)
			pred = test_out.max(-1)[1].detach().cpu().numpy()
			score = test_out.detach().cpu().numpy()
			if total_pred is None:
				total_pred = pred
				total_score = score
			else:
				total_pred = np.concatenate([total_pred, pred])
				total_score = np.concatenate([total_score, score])
	submit = np.eye(17)[total_pred.reshape(-1)]
	np.savetxt('./submit/prediction_' + str(best_score) + '.csv', submit, delimiter=',', fmt='%d')
	np.savetxt('./score/pre_score_' + str(best_score) + '.csv', total_score, delimiter=',', fmt='%.5f')
	print('completed!')
	print('-' * 80)
