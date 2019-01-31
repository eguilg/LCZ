from config import *
import h5py
import numpy as np


test_a_score_path = osp.join(results_root, 'score/integrate_testa_softmax.csv')
test_b_score_path = osp.join(results_root, 'score_B/integrate_softmax_2.csv')

test_a_score = np.loadtxt(test_a_score_path, delimiter=',')
test_b_score = np.loadtxt(test_b_score_path, delimiter=',')

test_a_h5 = h5py.File(testA_file, 'r')
test_b_h5 = h5py.File(testB_file, 'r')

soft_label_dataset = h5py.File(soft_labeld_data_file, 'a')


sen1 = np.concatenate([test_a_h5['sen1'],test_b_h5['sen1']], axis=0)
sen2 = np.concatenate([test_a_h5['sen2'],test_b_h5['sen2']], axis=0)
label = np.concatenate([test_a_score, test_b_score], axis=0)

soft_label_dataset.create_dataset('sen1', data=sen1)
soft_label_dataset.create_dataset('sen2', data=sen2)
soft_label_dataset.create_dataset('label', data=label)


test_a_h5.close()
test_b_h5.close()
soft_label_dataset.close()