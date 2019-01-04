import os
import h5py
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from dataloader import MyDataLoader, H5DataSource
from preprocess import prepare_batch
from modules.gac_net import GACNet
from modules.lcz_res_net import resnet18, resnet34, resnet50
from modules.lcz_dense_net import densenet121, densenet169, densenet201, densenet161

import torchvision.models as models

BATCH_SIZE = 32
N_CHANNEL = 26
test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std_f_test.h5'

MODEL = 'GAC'
# MODEL = 'DENSE'
# MODEL = 'RES'
# MODEL = 'LCZ'


model_dir = './checkpoints/model_93071'  # GACNet cosine GP  L2 3e-2 trained on train val 1:1  0.9046 0.852/0.8729 0.833
model_dir = './checkpoints/model_12224'  # GACNet cosine GP  L2 1e-2 trained on train val 1:1  0.9453 0.807
model_dir = './checkpoints/model_10774'  # GACNet cosine GP  L2 MIXUP 1.5e-2 trained on train val 1:1  0.89417 0.809
model_dir = './checkpoints/model_83173'  # GACNet cosine GP  L2 MIXUP 1e-2 trained on train val 1:1  0.9046 ?

cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if not os.path.isdir('./submit/'):
	os.mkdir('./submit/')
if not os.path.isdir('./score/'):
	os.mkdir('./score/')
if __name__ == '__main__':

	mean_std_h5 = h5py.File(mean_std_file, 'r')
	N_CHANNEL = mean_std_h5['mean'].shape[-1]
	mean = [torch.from_numpy(np.array(mean_std_h5['mean']).reshape(-1, N_CHANNEL).mean(0)).cuda()]
	std = [torch.from_numpy(np.sqrt((np.array(mean_std_h5['std']).reshape(-1, N_CHANNEL) ** 2).mean(0))).cuda()]
	# mean = torch.from_numpy(np.array(mean_std_h5['mean'])).float().cuda()
	# std = torch.from_numpy(np.array(mean_std_h5['std'])).float().cuda()
	mean_std_h5.close()

	mean, std = None, None

	data_source = H5DataSource([test_file], BATCH_SIZE, shuffle=False)
	test_loader = MyDataLoader(data_source.h5fids, data_source.indices)

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
	print('Testing...')
	total_pred = None
	total_score = None
	with torch.no_grad():
		model.eval()
		for test_data, _, fidx in tqdm(test_loader):
			test_input, _ = prepare_batch(test_data, None, fidx, mean, std)
			test_out = F.softmax(model(test_input))
			pred = test_out.max(-1)[1].detach().cpu().numpy()
			score = test_out.detach().cpu().numpy()
			if total_pred is None:
				total_pred = pred
				total_score = score
			else:
				total_pred = np.concatenate([total_pred, pred])
				total_score = np.concatenate([total_score, score])
	submit = np.eye(17)[total_pred.reshape(-1)]
	np.savetxt('./submit/sub_' + str(best_score) + '.csv', submit, delimiter=',', fmt='%d')
	np.savetxt('./score/score_' + str(best_score) + '.csv', total_score, delimiter=',', fmt='%.5f')
	print('completed!')
	print('-' * 80)
