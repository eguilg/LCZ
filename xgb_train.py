import xgboost as xgb
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_train.csv'
dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_val.csv'
dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_test.csv'

NUM_ROUNDS = 100000
SEED = 502
FOLD = 3
GABOR=False
extra  = ''

#  resample strategy
VAL_RESAMPLE_TIMES = 10


if GABOR:
	dense_train_file = '/home/zydq/Datasets/LCZ/dense_f_gabor_train.csv'
	dense_val_file = '/home/zydq/Datasets/LCZ/dense_f_gabor_val.csv'
	dense_test_file = '/home/zydq/Datasets/LCZ/dense_f_gabor_test.csv'
	extra = 'gabor'

base_dir = './xgb_ckp_' + extra +str(FOLD)+ 'fold/'
if not os.path.isdir(base_dir):
	os.mkdir(base_dir)



def gen_resample_dict(Y, times):
	label, count = np.unique(Y, return_counts=True)
	return dict(zip(label, (times * count).astype(int)))


if __name__ == '__main__':
	train_df = pd.read_csv(dense_train_file)
	val_df = pd.read_csv(dense_val_file)
	test_df = pd.read_csv(dense_test_file)
	feat_names = train_df.columns.tolist()[:-1]

	train_X = train_df.values[:, :-1]
	train_Y = train_df.values[:, -1]

	val_X = val_df.values[:, :-1]
	val_Y = val_df.values[:, -1]
	val_idx = list(range(val_Y.shape[0]))

	test_X = test_df.values

	del train_df, val_df, test_df

	# SMOTE
	sm = SMOTE(random_state=SEED, sampling_strategy=lambda y: gen_resample_dict(y, VAL_RESAMPLE_TIMES))

	d_test = xgb.DMatrix(test_X, feature_names=feat_names)
	test_pred_total = None
	val_pred_total = None
	y_total = None
	# trainval_Y = np.eye(17)[trainval_df['label'].astype(int).values.reshape(-1)]

	kf = KFold(n_splits=FOLD, random_state=SEED, shuffle=True)
	# fold_xgb = []
	folds_stack_x = np.array([])
	folds_stack_y = np.array([])
	for foldid, (train_index, val_index) in enumerate(kf.split(train_X)):
		print('=' * 80)
		print('starting fold: ', foldid)
		print('=' * 80)
		np.random.seed(SEED)
		np.random.shuffle(val_idx)
		val_X = val_X[val_idx]
		val_Y = val_Y[val_idx]
		val_res_X, val_res_Y = sm.fit_resample(val_X, val_Y)

		# train on big one
		x_train, x_val = train_X[train_index], train_X[val_index]
		y_train, y_val = train_Y[train_index], train_Y[val_index]

		# train on small one
		# x_val, x_train = train_X[train_index], train_X[val_index]
		# y_val, y_train = train_Y[train_index], train_Y[val_index]

		x_train = np.concatenate([x_train, val_res_X], axis=0)
		y_train = np.concatenate([y_train, val_res_Y], axis=0)

		# shuffle
		train_shuffled_idx = list(range(x_train.shape[0]))
		np.random.seed(SEED)
		np.random.shuffle(train_shuffled_idx)
		x_train, y_train = x_train[train_shuffled_idx], y_train[train_shuffled_idx]

		print((x_train.shape))
		params = {
			'tree_method': 'gpu_hist',
			'booster': 'gbtree',
			'objective': 'multi:softprob',  # 多分类的问题
			'eval_metric': 'merror',
			'num_class': 17,  # 类别数，与 multisoftmax 并用
			'gamma': 2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
			'max_depth': 4,  # 构建树的深度，越大越容易过拟合
			'n_estimators': 15,
			'lambda': 8,  # L2
			'alpha': 8, # L1
			'subsample': 0.5,  # 随机采样训练样本
			'colsample_by*': 0.5,  # 生成树时进行的列采样
			'min_child_weight': 5,
			'scale_pos_weight': 5,
			'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
			'eta': 0.02,  # 如同学习率
			'seed': 502,
			# 'nthread': 4,  # cpu 线程数
		}

		d_train = xgb.DMatrix(x_train, label=y_train, feature_names=feat_names)
		d_val = xgb.DMatrix(x_val, label=y_val, feature_names=feat_names)
		watchlist = [(d_train, 'train'), (d_val, 'valid')]
		bst_xgb = xgb.train(params, d_train, NUM_ROUNDS, watchlist, early_stopping_rounds=10, verbose_eval=10)
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

		test_pred = bst_xgb.predict(d_test).reshape(test_X.shape[0], 17, 1)

		if test_pred_total is None:
			test_pred_total = test_pred
		else:
			test_pred_total = np.concatenate([test_pred_total, test_pred], axis=-1)

		fold_submit = np.eye(17)[test_pred.argmax(1).reshape(-1)]
		np.savetxt('./submit/sub_xgb_'+ extra +str(FOLD)+ 'fold' + str(foldid) + '_' + str(val_score) + '.csv', fold_submit, delimiter=',', fmt='%d')
		np.savetxt('./score/score_xgb_'+ extra +str(FOLD)+ 'fold' + str(foldid) + '_' + str(val_score) + '.csv', test_pred[:, :, 0], delimiter=',', fmt='%.5f')

		bst_xgb.save_model(os.path.join(base_dir, 'xgb_'+ extra +str(FOLD)+ 'fold' + str(foldid) + '.model'))
		bst_xgb.dump_model(os.path.join(base_dir, 'xgb_'+ extra +str(FOLD)+ 'fold' + str(foldid) + '.dump.raw'))
		bst_xgb.__del__()
		d_train.__del__()
		d_val.__del__()
		del d_train, d_val, x_train, x_val, y_train, y_val, val_res_X, val_res_Y

	total_score = (np.argmax(val_pred_total, -1) == y_total).sum() / y_total.shape[0]
	print("total val score :", total_score)
	test_pred_total = test_pred_total.mean(axis=-1, keepdims=False)

	submit = np.eye(17)[test_pred_total.argmax(-1).reshape(-1)]
	np.savetxt('./submit/sub_xgb_' + extra + str(FOLD)+ 'full_' + str(total_score) + '.csv', submit, delimiter=',', fmt='%d')
	np.savetxt('./score/score_xgb_' + extra + str(FOLD)+ 'full_' + str(total_score) + '.csv', test_pred_total, delimiter=',', fmt='%.5f')
