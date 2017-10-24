import numpy as np
import random
import sys
import multiprocessing
from time import time	
import pyanitools as pya
import math




def data():

	hdf5file = 'ani_gdb_s01.h5'
	adl = pya.anidataloader(hdf5file)
	coordinates,output_training = extract_mol(adl)
	
	adl.cleanup()
	return coordinates,output_training

def extract_mol(adl):
	i = 0
	contador = 0
	for data in adl:
		
		if i == 2:
		    # Extract the data
		    P = data['path']
		    X = data['coordinates']
		    E = data['energies']
		    S = data['species']

		    # Print the data
		    #print("Path:   ", P)
		    #print("  Symbols:     ", S)
		    #print("  Coordinates: ", X)
		    #print("  Energies:    ", E,"\n")
		    
		i += 1
	
	return X,E



def bb(big_batch,data_input):
	print("entre")

	global n 


	for i in range(n):

		
		big_batch.put(data_input[random.randint(0,len(data_input)-1)])
		
	print("sali")


	

def generator(big_batch):
	print("entre2")
	global n
	mini_batch_size = 100
	iterations = int(big_batch.qsize() / mini_batch_size)
	
	for i in range(iterations):

		for j in range(mini_batch_size):

			yield big_batch.get()





if __name__ == "__main__":


	t1 = time()
	coordinates,output_training = data()
	n = 1000
	big_batch = multiprocessing.Queue()
	
	p1 = multiprocessing.Process(target=bb,args=(big_batch,coordinates))
	p2 = multiprocessing.Process(target=generator,args=(big_batch,))


	
	generator = generator(big_batch)


	p1.start()
	p2.start()
	
	p2.join()
	
	
	for i in generator:
		print(i)
	t2 = time()
	
	tt = t2 - t1
	print(tt)

	
	t3 = time()

	
	