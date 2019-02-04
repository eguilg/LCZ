import os
import numpy as np
from config import *
import os.path as osp

input_dir = score_dir
inputs = [
	'DENSE121_mixup0_foc1_weight0_decay0.01.csv',
	'DENSE201_mixup0_foc1_weight0_decay0.01.csv',
	'GAC_mixup0_foc1_weight0_decay0.01.csv',
	'RES10_mixup0_foc0_weight1_decay0.01.csv',
	'RES10_mixup0_foc1_weight0_decay0.01.csv',
	'RES10_mixup0_foc1_weight0_decay0.01_crop.csv',
	'RES10_mixup0_foc1_weight0_decay0.01_crop_1:7.csv',
	'RES18_mixup0_foc1_weight0_decay0.01.csv',
	'SE-RES10_mixup0_foc1_weight0_decay0.01.csv',
	'SE-RES15_mixup0_foc1_weight0_decay0.01.csv',
	'XCEPTION_mixup0_foc1_weight0_decay0.01.csv',

	'score_0.9046128156960969.csv',
	'score_0.9307034022124817.csv',
	'score_0.9163013984554372.csv',

]

inputs = [os.path.join(input_dir, input) for input in inputs]

print(inputs)

# 84.5
# inputs = [
# 	'pre_score_0.9765625.csv',
# 	'pre_score_0.9861111111111112.csv',
# ]

out = 0
total_score = 0
for path in inputs:

	# score = float(os.path.splitext(input)[0].split('_')[-1])
	score = 1
	preds = np.loadtxt(path, delimiter=',')
	out += score * preds
	total_score += score

out /= total_score
submit = np.eye(17)[np.argmax(out, axis=-1).reshape(-1)]

np.savetxt(osp.join(submit_dir, 'ensemble' + '.csv'), submit, delimiter=',', fmt='%d')
np.savetxt(osp.join(score_dir, 'ensemble' + '.csv'), out, delimiter=',', fmt='%.5f')
