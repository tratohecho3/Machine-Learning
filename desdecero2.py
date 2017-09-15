import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pyanitools as pya
import math
import torch.nn.functional as F
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

def combinations(coordinates,energies):
	data = []
	output = []
	counter = 0
	for matrix in coordinates:
		vector = []
		for i in range(3):
			
			if i == 0: 
				dist1 = distance(matrix[i],matrix[i+1])
				dist2 = distance(matrix[i],matrix[i+2])
				vector.append(dist1)
				vector.append(dist2)	
				
			if i == 1:
				dist1 = distance(matrix[i],matrix[i+1])
				vector.append(dist1)
		data.append(vector)
		output.append(energies[counter])

		counter += 1
	return data,output

def distance(vector1,vector2):
	x = vector1[0] - vector2[0]
	y = vector1[1] - vector2[1]
	z = vector1[2] - vector2[2]
	distance = math.sqrt(x*x + y*y + z*z)

	return distance

hdf5file = 'gdb11_S01_06r.h5'
adl = pya.anidataloader(hdf5file)
coordinates = extract_mol(adl)[0]
energies = extract_mol(adl)[1]
data = combinations(coordinates,energies)[0]
output = combinations(coordinates,energies)[1]
adl.cleanup()

#DATA SET
training_set,test_set,output_training,output_test = ten_percent(data,output)

def rellenar(x):
	for i in range(len(x)):
		x[i] = [x[i][0],x[i][0],x[i][0]]
	return x
"""
training_set = [[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]]
output_training = [[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]]
training_set = rellenar(training_set)
"""

idx = np.arange(float(y.size()[0]))

bs = 100


#PARAMETERS
input_size = 3
output_size = 1
hidden_size = 8
num_epochs = 750
learning_rate = 0.01

#NEURAL NETWORK
class LinearRegression(nn.Module):
	def __init__(self,input_size,output_size,hidden_size):
		super(LinearRegression,self).__init__()
		self.linear = nn.Linear(input_size,output_size)
		#self.linear2 = nn.Linear(hidden_size,output_size)
		

	def forward(self,x):
		out = self.linear(x)
		#out = F.tanh(out)
		#out = self.linear2(out)

		return out




#MODEL,LOSS FUNCTION,OPTIMIZER
model = LinearRegression(input_size,output_size,hidden_size)		
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


#TRAINING
for epoch in range(num_epochs):
	for b in range(int(np.floor(y.size()[0]/bs))):
		bidx = Variable(torch.FloatTensor(idx[b*bs:(b+1)*bs]),requires_grad=False)
		if bidx.data.numpy().shape != (0,):
			new_input = Variable(torch.FloatTensor(x.data.numpy()[bidx.data.numpy().astype(int)]),requires_grad=False)
		x = Variable(torch.FloatTensor(training_set))
		y = Variable(torch.FloatTensor(output_training))

		#FORWARD,BACKWARD,OPTIMIZER
		
		y_pred = model(x)
		loss = criterion(y_pred,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(epoch,loss.data[0])

x_test = Variable(torch.FloatTensor(training_set))
predicted = model(x_test)

"""
plt.plot(output_training, output_training, 'ro', label='Original data')
#plt.plot(training_set, predicted, label='Fitted line')
plt.legend()
plt.show()
"""


"""
plt.plot(y.data.numpy(), x.data.numpy())
plt.scatter(y.data.numpy(),predicted.data.numpy())
plt.show()
"""
#print(y_pred.data.numpy())




"""
plt.plot(output_training,output_training)
plt.scatter(output_training,y_pred.data.numpy())
plt.show()
"""
print(output_training[800:810])
print('')
print(predicted.data.numpy()[800:810])

#plt.plot([np.asarray(output_training).min(),np.asarray(output_training).max()],[np.asarray(output_training).min(),np.asarray(output_training).max()])
plt.scatter(output_training,predicted.data.numpy())
plt.show()
