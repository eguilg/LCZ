import os.path as osp

LOCAL = True
TEST_B = False

if LOCAL:
	model_root = './checkpoints'
	# data_root = '/home/zydq/Datasets/LCZ/'
	data_root = '/data/pzq/tianchi/data/'
	results_root = './results'
else:
	model_root = '../checkpoints'
	data_root = '/nas/LCZ/'
	results_root = '../results/'

train_file = osp.join(data_root, 'training.h5')
val_file = osp.join(data_root, 'validation.h5')
soft_labeld_data_file = osp.join(data_root, 'soft_labeled_data_3t.h5')
soft_a_path = osp.join(data_root, 'soft_a.h5')
soft_b_path = osp.join(data_root, 'soft_b.h5')
soft_2a_path = osp.join(data_root, 'soft_2a.h5')
soft_2b_path = osp.join(data_root, 'soft_2b.h5')

testA_file = osp.join(data_root, 'round1_test_a_20181109.h5')
testB_file = osp.join(data_root, 'round1_test_b_20190104.h5')  # 1B榜
test2A_file = osp.join(data_root, 'round2_test_a_20190121.h5')
test2B_file = osp.join(data_root, 'round2_test_b_20190211.h5')  # 2B榜

mean_std_train_file = osp.join(data_root, 'mean_std_train.h5')
mean_std_val_file = osp.join(data_root, 'mean_std_val.h5')
mean_std_soft_label_file = osp.join(data_root, 'mean_std_soft.h5')
mean_std_test2a_file = osp.join(data_root, 'mean_std_test2a.h5')
mean_std_test2b_file = osp.join(data_root, 'mean_std_test2b.h5')


if TEST_B:
	test_file = test2B_file
	mean_std_test_file = mean_std_test2b_file
	submit_dir = osp.join(results_root, 'submit2_B')
	score_dir = osp.join(results_root, 'score2_B')
else:
	test_file = test2A_file
	mean_std_test_file = mean_std_test2a_file
	submit_dir = osp.join(results_root, 'submit2_A')
	score_dir = osp.join(results_root, 'score2_A')

SEMI_SPV = True
ZSCORE = False
CROP_CUTOUT = False
USE_CLASS_WEIGHT = False
MIX_UP = False
FOCAL = False
SOFT = True
FINE_TUNE = False


SEED = 502
EPOCH = 13
BATCH_SIZE = 64
MIX_UP_ALPHA = 1.0
N_CHANNEL = 26

T = 1.5
ROUND = 6
EPOCH = int(T * ROUND)
N_SNAPSHOT = 6

LR = 0.0001
DECAY = 1e-2
L1_WEIGHT = 0

# MODEL = 'GAC'
MODEL = 'RES10'
# MODEL = 'RESW10'
# MODEL = 'RES18'
# MODEL = 'SE-RES10'
# MODEL = 'SE-RES15'
# MODEL = 'SE-RES-YS'
# MODEL = 'RESNEXT'
# MODEL = 'DENSE121'
# MODEL = 'DENSE201'
# MODEL = 'DENSE-YS'
# MODEL = 'XCEPTION'

name_arg = [MODEL,
			'lr' + str(LR),
			'bs'+str(BATCH_SIZE),
			'l1_' + str(L1_WEIGHT),
			'l2_' + str(DECAY),
			'T' + str(T)
			]

# extra_name = ['onval']
extra_name = ['semi4t_lam0']
SCORE_THRESH = 0.89
TEST_REPEAT = 10
name_arg += extra_name
model_name = '_'.join(name_arg)

# model_name = 'model_93071'; MODEL = 'GAC'# GACNet cosine GP  L2 3e-2 trained on train val 1:1  0.9046 A0.852/0.8729 A0.833
# model_name = 'model_83173'; MODEL = 'GAC'  # GACNet cosine GP  L2 1e-2 MIXUP trained on train val 1:1  0.9163 A0.837
# model_name = 'model_79740'; MODEL = 'GAC'  # GACNet cosine GP  L2 1.5e-2 FOCAL trained on train val 1:1 0.9307


# model_name = 'GAC_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi2t_th0.8_lam0' # na
# model_name = 'GAC_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_lam0'
# model_name = 'GAC_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_onehot_lam0'
# model_name = 'RES10_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi2t_th0.8_lam0'
# model_name = 'RES10_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_lam0'
# model_name = 'RES10_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_onehot_lam0'
# model_name = 'RES18_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi2t_th0.8_lam0'
# model_name = 'RES18_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_lam0'
# model_name = 'RES18_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_onehot_lam0'
# model_name = 'RESNEXT_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi2t_th0.8_lam0'
# model_name = 'RESNEXT_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_lam0'
# model_name = 'RESNEXT_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi3t_onehot_lam0'

# model_name = 'RES10_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi4t_lam0'
# model_name = 'RES18_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi4t_lam0'
# model_name = 'RESNEXT_lr0.0001_bs64_l1_0_l2_0.01_T1.5_semi4t_lam0'