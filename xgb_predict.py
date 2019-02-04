import xgboost as xgb
import numpy as np
import pandas as pd
import os

# dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_train.csv'
# dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_val.csv'
TEST_B = False

dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_test2A.csv'
submit_dir = './submit2_A/'
score_dir = './score2_A/'
if TEST_B:
	dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_test2B.csv'
	submit_dir = './submit2_B/'
	score_dir = './score2_B/'


if not os.path.isdir(submit_dir):
	os.mkdir(submit_dir)
if not os.path.isdir(score_dir):
	os.mkdir(score_dir)

model_paths = [
	'./xgb_ckp_3fold/xgb_3fold0.model',
	'./xgb_ckp_3fold/xgb_3fold1.model',
	'./xgb_ckp_3fold/xgb_3fold2.model',
]


if __name__ == '__main__':
	# train_df = pd.read_csv(dense_train_file)
	# val_df = pd.read_csv(dense_val_file)
	test_df = pd.read_csv(dense_test_file)
	d_test = xgb.DMatrix(test_df)
	pred_total = None
	for path in model_paths:
		xgb_model = xgb.Booster({'nthread': 4})  # init model
		xgb_model.load_model(path)  # load data
		# xgb_model = xgb.load_model(path)

		pred = xgb_model.predict(d_test).reshape(test_df.shape[0], 17, 1)

		if pred_total is None:
			pred_total = pred
		else:
			pred_total = np.concatenate([pred_total, pred], -1)

	pred_total = pred_total.mean(-1)
	submit = np.eye(17)[pred_total.argmax(-1).reshape(-1)]
	np.savetxt(os.path.join(submit_dir, 'sub_xgb_full' + '.csv'), submit, delimiter=',', fmt='%d')
	np.savetxt(os.path.join(score_dir, 'score_xgb_full' + '.csv'), pred_total, delimiter=',', fmt='%.5f')

