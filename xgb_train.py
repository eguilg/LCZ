import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_train.csv'
dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_val.csv'
dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_test.csv'
if not os.path.isdir('./xgb_ckp/'):
	os.mkdir('./xgb_ckp/')

NUM_ROUNDS = 100000
SEED = 502

#  resample strategy
VAL_RESAMPLE_TIMES = 10

def gen_resample_dict(Y):
	label, count = np.unique(Y, return_counts=True)
	return dict(zip(label, VAL_RESAMPLE_TIMES * count))


if __name__ == '__main__':
	train_df = pd.read_csv(dense_train_file)
	val_df = pd.read_csv(dense_val_file)

	test_df = pd.read_csv(dense_test_file)

	train_X = train_df.iloc[:, :-1]
	train_Y = train_df['label']

	val_X = val_df.iloc[:, :-1]
	val_Y = val_df['label']

	# SMOTE
	sm = SMOTE(random_state=SEED, sampling_strategy=gen_resample_dict)
	val_res_X, val_res_Y = sm.fit_resample(val_X, val_Y)
	val_res_X = pd.DataFrame(val_res_X, columns=val_X.columns.tolist())
	val_res_Y = pd.Series(val_res_Y, name='label')

	# trainval_df = pd.concat([train_df, val_df], axis=0)
	# trainval_X = trainval_df.iloc[:, :-1]
	# trainval_Y = trainval_df['label']

	d_test = xgb.DMatrix(test_df)
	test_pred_total = None
	val_pred_total = None
	y_total = None
	# trainval_Y = np.eye(17)[trainval_df['label'].astype(int).values.reshape(-1)]

	kf = KFold(n_splits=5, random_state=502, shuffle=True)
	# fold_xgb = []
	folds_stack_x = np.array([])
	folds_stack_y = np.array([])
	for foldid, (train_index, val_index) in enumerate(kf.split(train_df.values)):

		x_train, x_val = train_X.iloc[train_index], train_X.iloc[val_index]
		y_train, y_val = train_Y.iloc[train_index], train_Y.iloc[val_index]

		x_train = pd.concat([x_train, val_res_X], axis=0)
		y_train = pd.concat([y_train, val_res_Y], axis=0)

		# shuffle
		np.random.seed(SEED)
		train_shuffled_idx = np.random.shuffle(list(range(x_train.shape[0])))
		x_train, y_train = x_train.iloc[train_shuffled_idx], y_train.iloc[train_shuffled_idx]

		print((x_train.shape))
		params = {
			'tree_method': 'gpu_hist',
			'booster': 'gbtree',
			'objective': 'multi:softprob',  # 多分类的问题
			'eval_metric': 'merror',
			'num_class': 17,  # 类别数，与 multisoftmax 并用
			'gamma': 0.20,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
			'max_depth': 12,  # 构建树的深度，越大越容易过拟合
			'lambda': 2.5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
			'subsample': 0.5,  # 随机采样训练样本
			'colsample_bytree': 0.7,  # 生成树时进行的列采样
			'min_child_weight': 3,
			'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
			'eta': 0.01,  # 如同学习率
			'seed': 502,
			# 'nthread': 4,  # cpu 线程数
		}

		d_train = xgb.DMatrix(x_train, label=y_train)
		d_val = xgb.DMatrix(x_val, label=y_val)
		watchlist = [(d_train, 'train'), (d_val, 'valid')]
		bst_xgb = xgb.train(params, d_train, NUM_ROUNDS, watchlist, early_stopping_rounds=15, verbose_eval=10)
		# fold_xgb.append(bst_xgb)
		val_pred = bst_xgb.predict(d_val).reshape(y_val.shape[0], 17)
		val_score = (np.argmax(val_pred, -1) == y_val).sum() / y_val.shape[0]
		print("val score :", val_score)

		if val_pred_total is None:
			val_pred_total = val_pred
			y_total = y_val
		else:
			val_pred_total = np.concatenate([val_pred_total, val_pred], axis=0)
			y_total = np.concatenate([y_total, y_val], axis=0)

		test_pred = bst_xgb.predict(d_test).reshape(test_df.shape[0], 17, 1)

		if test_pred_total is None:
			test_pred_total = test_pred
		else:
			test_pred_total = np.concatenate([test_pred_total, test_pred], axis=-1)

		bst_xgb.save_model('./xgb_ckp/xgb_fold' + str(foldid) + '.model')
		bst_xgb.dump_model('./xgb_ckp/xgb_fold' + str(foldid) + '.xgbfi')
		bst_xgb.__del__()
		del d_train, d_val, x_train, x_val, y_train, y_val

	total_score = (np.argmax(val_pred_total, -1) == y_total).sum() / y_total.shape[0]
	print("total val score :", total_score)
	test_pred_total = test_pred_total.mean(axis=-1, keepdims=False)

	submit = np.eye(17)[test_pred_total.argmax(-1).reshape(-1)]
	np.savetxt('./submit/prediction_xgb_' + str(total_score) + '.csv', submit, delimiter=',', fmt='%d')
	np.savetxt('./score/pre_score_xgb_' + str(total_score) + '.csv', test_pred_total, delimiter=',', fmt='%.5f')
