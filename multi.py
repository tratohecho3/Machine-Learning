import multiprocessing
import os
def worker(number):
	print("Mi funcion imprime",number,os.getpid())
	return


if __name__ == '__main__':
	jobs = []
	for i in range(5):
		p = multiprocessing.Process(target=worker,args=(i,))
		jobs.append(p)
		p.start()

	p.join()
	print(jobs)
