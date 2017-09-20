import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pyanitools as pya
import math
import torch.nn.functional as F
import torch.nn.init as init

def ten_percent(data,output):
	ten_percent = int(len(data) * 0.1)	
	training_set = data[ten_percent:]
	test_set  = data[:ten_percent]
	output_training = output[ten_percent:]
	output_test = output[:ten_percent]

	return training_set,test_set,output_training,output_test

def changeHA(x):
	final = x * 627.509469

	return final

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


def combinations(coordinates):
	training_set = []
	
	for matrix in coordinates:
	
		distance1 = distance(matrix[0],matrix[1])
		distance2 = distance(matrix[0],matrix[2])
		distance3 = distance(matrix[1],matrix[2])
		training_set.append([distance1,distance2,distance3])
	return training_set

def distance(vector1,vector2):
	x = vector1[0] - vector2[0]
	y = vector1[1] - vector2[1]
	z = vector1[2] - vector2[2]
	distance = math.sqrt(x*x + y*y + z*z)

	return distance


#NEURAL NETWORK
class LinearRegression(nn.Module):
	def __init__(self,input_size,output_size,hidden_size):
		super(LinearRegression,self).__init__()
		self.linear = nn.Linear(input_size,hidden_size)
		init.xavier_uniform(self.linear.weight, gain=np.sqrt(2.0))
		init.constant(self.linear.bias, 0)
		self.linear2 = nn.Linear(hidden_size,output_size)
		init.xavier_uniform(self.linear2.weight, gain=np.sqrt(2.0))
		init.constant(self.linear2.bias, 0)
		

	def forward(self,x):
		out = self.linear(x)
		out = F.relu(out)
		out = self.linear2(out)

		return out



hdf5file = 'gdb11_S01_06r.h5'
adl = pya.anidataloader(hdf5file)
coordinates,output_training = extract_mol(adl)
training_set = combinations(coordinates)
adl.cleanup()


#DATA SET
training_set,test_set,output_training,output_test = ten_percent(training_set,output_training)


#PARAMETERS
input_size = 3
output_size = 1
hidden_size = 6
num_epochs = 5000
learning_rate = 0.01

#MODEL,LOSS FUNCTION,OPTIMIZER
model = LinearRegression(input_size,output_size,hidden_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)



#TRAINING
for epoch in range(num_epochs):
	
	x = Variable(torch.FloatTensor(training_set),requires_grad=True)

	y = Variable(torch.FloatTensor(output_training))

	#FORWARD,BACKWARD,OPTIMIZER
	
	y_pred = model(x)
		
	loss = criterion(y_pred,y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print(epoch,loss.data[0])
	

x_test = Variable(torch.FloatTensor(test_set))
predicted = model(x_test)


#plt.plot([np.asarray(output_training).min(),np.asarray(output_training).max()],[np.asarray(output_training).min(),np.asarray(output_training).max()])
plt.scatter(output_test,[predicted.data.numpy()])
plt.show()
