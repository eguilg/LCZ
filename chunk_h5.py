import h5py
from tqdm import tqdm

f_org = h5py.File('/home/zydq/Datasets/LCZ/training.h5', 'r')
f_chunk = h5py.File('/home/zydq/Datasets/LCZ/training_chunked.h5', 'a')

f_chunk.create_dataset('label', f_org['label'].shape, compression="gzip")
f_chunk.create_dataset('sen1', f_org['sen1'].shape, compression="gzip")
f_chunk.create_dataset('sen2', f_org['sen2'].shape, compression="gzip")
print(f_chunk['label'].chunks, f_chunk['sen1'].chunks, f_chunk['sen2'].chunks)

buffer_size = 10000

for i in tqdm(range(int(f_org['sen1'].shape[0] / buffer_size) + 1)):
	if (i+1)*buffer_size < f_org['sen1'].shape[0]:
		indices = list(range(i*buffer_size, (i+1)*buffer_size))
	else:
		indices = list(range(i*buffer_size, f_org['sen1'].shape[0]))
	f_chunk['label'][indices] = f_org['label'][indices]
	f_chunk['sen1'][indices] = f_org['sen1'][indices]
	f_chunk['sen2'][indices] = f_org['sen2'][indices]

f_org.close()
f_chunk.close()