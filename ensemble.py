import os
import numpy as np

input_dir = './score'
inputs = [
	'score_0.9738586523125997.csv',  # 0.804
	'score_0.9744318181818182.csv',  # 0.807
	'score_0.9046128156960969.csv',
	'score_xgb_3full_0.9472395180011692.csv',  # 不知道
	'score_xgb_4full_0.9358961988387074.csv',  # 不知道
	'score_xgb_full_0.9702184660268017.csv'  # 0.816 以上
]

llx_input_dir = './score_llx'
llx_inputs = [
	'test819_batch64_steps800_lczNetNew_data_softmax.csv',
	'sen2_renext_softmax.csv',
	'resnextNew_data2_softmax.csv',
	'resnext_four_train985_val945_softmax.csv'
]

inputs = [os.path.join(input_dir, input) for input in inputs]
llx_inputs = [os.path.join(llx_input_dir, input) for input in llx_inputs]

inputs += llx_inputs
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
np.savetxt('./submit/ensemble' + '.csv', submit, delimiter=',', fmt='%d')
