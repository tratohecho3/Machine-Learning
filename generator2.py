import numpy as np
import random
import sys
import multiprocessing
from time import time	

def bb():
	global n
	big_batch = np.random.random_sample((n))
	return big_batch
	

	

def generator(big_batch):
	global n
	mini_batch_size = 100
	iterations = int(len(big_batch) / mini_batch_size)
	
	for i in range(iterations):
		for i in range(mini_batch_size):

			yield big_batch[random.randint(0,n-1)]





if __name__ == "__main__":

	t1 = time()
	n = 1000000
	big_batch = bb()
	generator = generator(big_batch)
	for i in generator:
		print(i)
	t2 = time()
	
	tt = t2 - t1
	print(tt)


	

	
