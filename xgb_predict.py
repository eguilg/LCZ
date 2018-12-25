import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold

# dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_train.csv'
# dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_val.csv'
dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_test.csv'
model_paths = [
	'./xgb_ckp/xgb_fold0.model',
	# './xgb_ckp/xgb_fold1.model',

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
	np.savetxt('./submit/sub_xgb_full' + '.csv', submit, delimiter=',', fmt='%d')
	np.savetxt('./score/score_xgb_full' + '.csv', pred_total, delimiter=',', fmt='%.5f')

