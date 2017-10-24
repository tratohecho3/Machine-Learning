from time import time
import h5py
import numpy as np

with h5py.File('tmp.h5', 'w') as f:
    f.create_dataset('data1', data=np.random.random((128*1024, 128, 4)).astype(np.float32), chunks=(1024, 1, 1))
    f.create_dataset('data2', data=np.random.random((128*1024, 32, 2)).astype(np.float32), chunks=(1024, 1, 1))

def generator():
    with h5py.File('tmp.h5', 'r') as f:
        random_chunks = np.arange(128)
        np.random.shuffle(random_chunks)
        for i in range(32): # 128 // 4
            selected_chunks = random_chunks[i*4:(i+1)*4]
            data1 = np.concatenate([f['data1'][i*1024:(i+1)*1024] for i in selected_chunks], axis=1)
            data2 = np.concatenate([f['data2'][i*1024:(i+1)*1024] for i in selected_chunks], axis=1)
            yield data1, data2
        
t1 = time()
[i for i in generator()]
time() - t1