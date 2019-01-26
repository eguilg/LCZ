import os.path as osp

LOCAL = False
TEST_B = False

if LOCAL:
	model_root = './checkpoints'
	data_root = '/home/zydq/Datasets/LCZ/'
	results_root = './'
else:
	model_root = '../checkpoints'
	data_root = '/nas/LCZ/'
	results_root = '../results/'

train_file = osp.join(data_root, 'training.h5')
val_file = osp.join(data_root, 'validation.h5')
test2A_file = osp.join(data_root, 'round2_test_a_20190121.h5')
test2B_file = osp.join(data_root, 'round2_test_b_20190121.h5')  # B榜
mean_std_train_file = osp.join(data_root, 'mean_std_train.h5')

mean_std_val_file = osp.join(data_root, 'mean_std_val.h5')
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



USE_CLASS_WEIGHT = False
MIX_UP = False
FOCAL = False
GHM = True
FINE_TUNE = False
TEST_REPEAT = 5

SEED = 502
BATCH_SIZE = 64
MIX_UP_ALPHA = 1.0
N_CHANNEL = 26

LR = 1e-4
DECAY = 1e-2

# MODEL = 'GAC'
MODEL = 'RES10'
# MODEL = 'RES18'
# MODEL = 'SE-RES10'
# MODEL = 'SE-RES15'
# MODEL = 'DENSE121'
# MODEL = 'DENSE201'
# MODEL = 'XCEPTION'

name_arg = [MODEL, 'mixup' + str(int(MIX_UP)), 'foc' + str(int(FOCAL)), 'weight' + str(int(USE_CLASS_WEIGHT)),
			'decay' + str(DECAY)]

# extra_name = ['onval']
extra_name = ['crop']

name_arg += extra_name
model_name = '_'.join(name_arg)

# model_name = 'model_93071'; MODEL = 'GAC'# GACNet cosine GP  L2 3e-2 trained on train val 1:1  0.9046 A0.852/0.8729 A0.833
# model_name = 'model_83173'; MODEL = 'GAC'  # GACNet cosine GP  L2 1e-2 MIXUP trained on train val 1:1  0.9163 A0.837
# model_name = 'model_79740'; MODEL = 'GAC'  # GACNet cosine GP  L2 1.5e-2 FOCAL trained on train val 1:1 0.9307
