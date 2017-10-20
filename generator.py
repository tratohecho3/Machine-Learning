import numpy as np
import random
import sys
import multiprocessing
from time import time	

def bb(big_batch):
	global n 

	for idx,number in enumerate(big_batch):

		big_batch[idx] = random.randint(1,n)

	

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
	big_batch = multiprocessing.Array('i',n)
	p1 = multiprocessing.Process(target=bb,args=(big_batch,))
	p2 = multiprocessing.Process(target=generator,args=(big_batch,))
	
	generator = generator(big_batch)


	p1.start()
	p2.start()
	p1.join()
	p2.join()
	
	for i in generator:
		print(i)
	t2 = time()
	
	tt = t2 - t1
	print(tt)


	t3 = time()


