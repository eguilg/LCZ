import os
import h5py
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from dataloader import MyDataLoader, H5DataSource
from preprocess import prepare_batch
from modules.gac_net import GACNet
from modules.lcz_xception import Xception
from modules.lcz_res_net import resnet10, resnet18, resnet34, resnet50
from modules.lcz_senet import se_resnet10_fc512, se_resnet15_fc512
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161

from config import  *

BATCH_SIZE = 100


model_dir = osp.join(model_root, model_name)

cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if not os.path.isdir(results_root):
	os.mkdir(results_root)
if not os.path.isdir(submit_dir):
	os.mkdir(submit_dir)
if not os.path.isdir(score_dir):
	os.mkdir(score_dir)
if __name__ == '__main__':

	mean_std_h5 = h5py.File(mean_std_test_file, 'r')
	mean = torch.from_numpy(np.array(mean_std_h5['mean'])).float().cuda()
	std = torch.from_numpy(np.array(mean_std_h5['std'])).float().cuda()
	mean_std_h5.close()


	data_source = H5DataSource([test_file], BATCH_SIZE, shuffle=False)
	test_loader = MyDataLoader(data_source.h5fids, data_source.indices)

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
		for test_data, _, fidx in tqdm(test_loader):
			test_input, _ = prepare_batch(test_data, None, fidx, mean, std)
			test_out = F.softmax(model(test_input), -1)
			pred = test_out.max(-1)[1].detach().cpu().numpy()
			score = test_out.detach().cpu().numpy()
			if total_pred is None:
				total_pred = pred
				total_score = score
			else:
				total_pred = np.concatenate([total_pred, pred])
				total_score = np.concatenate([total_score, score])
	submit = np.eye(17)[total_pred.reshape(-1)]
	np.savetxt(os.path.join(submit_dir, 'sub_' + str(best_score) + '.csv'), submit, delimiter=',', fmt='%d')
	np.savetxt(os.path.join(score_dir,'score_' + str(best_score) + '.csv'), total_score, delimiter=',', fmt='%.5f')
	print('completed!')
	print('-' * 80)
