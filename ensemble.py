import os
import numpy as np

input_dir = './score'
inputs = [
	'pre_score_0.8975694444444444.csv',
	'pre_score_0.9092881944444444.csv'
]

out = 0
total_score = 0
for input in inputs:
	path = os.path.join(input_dir, input)
	score = float(os.path.splitext(input)[0].split('_')[-1])
	preds = np.loadtxt(path, delimiter=',')
	out += score * preds
	total_score += score

out /= total_score
submit = np.eye(17)[np.argmax(out, axis=-1).reshape(-1)]
np.savetxt('./submit/ensemble' + '.csv', submit, delimiter=',', fmt='%d')
