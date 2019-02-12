from config import *
import h5py
import numpy as np


test_a_score_path = osp.join(results_root, 'score/esb_0.884_a.csv')
test_b_score_path = osp.join(results_root, 'score_B/esb_0.880_b.csv')
test_2a_score_path = osp.join(results_root, 'score2_A/esb_0.888_2a.csv')
test_2b_score_path = osp.join(results_root, 'score2_B/esb_0.892_2b.csv')



test_a_score = np.loadtxt(test_a_score_path, delimiter=',')
test_b_score = np.loadtxt(test_b_score_path, delimiter=',')
test_2a_score = np.loadtxt(test_2a_score_path, delimiter=',')
test_2b_score = np.loadtxt(test_2b_score_path, delimiter=',')


test_a_h5 = h5py.File(testA_file, 'r')
test_b_h5 = h5py.File(testB_file, 'r')
test_2a_h5 = h5py.File(test2A_file, 'r')
test_2b_h5 = h5py.File(test2B_file, 'r')

targets = [soft_a_path, soft_b_path, soft_2a_path, soft_2b_path]
source_h5 = [test_a_h5, test_b_h5, test_2a_h5, test_2b_h5]
source_label = [test_a_score, test_b_score, test_2a_score, test_2b_score]

# targets = [soft_2b_path]
# source_h5 = [test_2b_h5]
# source_label = [test_2b_score]

for i in range(len(targets)):

	soft_label_dataset = h5py.File(targets[i], 'a')
	sen1 = source_h5[i]['sen1']
	sen2 = source_h5[i]['sen2']
	label = source_label[i]

	soft_label_dataset.create_dataset('sen1', data=sen1)
	soft_label_dataset.create_dataset('sen2', data=sen2)
	soft_label_dataset.create_dataset('label', data=label)
	soft_label_dataset.close()
