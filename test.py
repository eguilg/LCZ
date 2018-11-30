import os
import time
import h5py
from tqdm import tqdm
import torch
import numpy as np
from dataloader import prepare_batch, MyDataLoader, H5DataSource
from model import LCZNet

# import torchvision.models as models

BATCH_SIZE = 32
test_file = '/home/zydq/Datasets/LCZ/round1_test_a_20181109.h5'
mean_std_file = '/home/zydq/Datasets/LCZ/mean_std.h5'

model_dir = './checkpoints/model_54017'
cur_model_path = os.path.join(model_dir, 'state_curr.ckpt')

if not os.path.isdir('./submit/'):
	os.mkdir('./submit/')
if __name__ == '__main__':

	mean_std_h5 = h5py.File(mean_std_file, 'r')
	mean = np.array(mean_std_h5['mean'])
	std = np.array(mean_std_h5['std'])
	mean_std_h5.close()

	data_source = H5DataSource([test_file], BATCH_SIZE, shuffle=False)
	test_loader = MyDataLoader(data_source.h5fids, data_source.indices)

	model = LCZNet(channel=18, n_class=17, base=64, dropout=0.3)
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
	with torch.no_grad():
		model.eval()
		for test_data, _ in tqdm(test_loader):
			test_input, _ = prepare_batch(test_data, None, mean, std)
			test_out = model(test_input)
			pred = test_out.max(-1)[1].detach().cpu().numpy()
			if total_pred is None:
				total_pred = pred
			else:
				total_pred = np.concatenate([total_pred, pred])
	submit = np.eye(17)[total_pred.reshape(-1)]
	np.savetxt('./submit/prediction_' + str(best_score) + '.csv', submit, delimiter=',', fmt='%d')
	print('completed!')
	print('-' * 80)
