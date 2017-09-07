import pyanitools as pya
import torch
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
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
	

# Set the HDF5 file containing the data
hdf5file = 'gdb11_S01_06r.h5'

# Construct the data loader class
adl = pya.anidataloader(hdf5file)

coordinates = extract_mol(adl)[0]
energies = extract_mol(adl)[1]
data = combinations(coordinates,energies)[0]
output = combinations(coordinates,energies)[1]


#final_data = []
#final_output = []
#final_data.append(data)
#final_output.append(output)

# Closes the H5 data file
adl.cleanup()

training_set,test_set,output_training,output_test = ten_percent(data,output)


x = Variable(torch.FloatTensor(training_set))
y = Variable(torch.FloatTensor(output_training),requires_grad=False)

model = torch.nn.Sequential(
	torch.nn.Linear(3,16),
	torch.nn.ReLU(),
	torch.nn.Linear(16,1),
        )


loss_fn = torch.nn.MSELoss(size_average=False)


learning_rate = 1e-4
coordx=[]
coordy=[]
for t in range(50000):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)

  print(t,loss.data[0])
  #print(t,math.log10(changeHA(loss.data[0])))
  coordx.append(t)

  coordy.append(loss.data[0])
  #coordy.append(math.log10(changeHA(loss.data[0])))
  
  model.zero_grad()
  loss.backward()

  for param in model.parameters():
    param.data -= learning_rate * param.grad.data


plt.plot(coordx,coordy)
plt.axis(ymin=0, ymax=3)
#plt.axis(ymin=0, ymax=15000)

#print(max(coordy),min(coordy))
#print(max(coordy) - min(coordy))
plt.show()


x = Variable(torch.FloatTensor(test_set))
y = Variable(torch.FloatTensor(output_test),requires_grad=False)

coordx=[]
coordy=[]
for t in range(50000):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)

  print(t,loss.data[0])
  #print(t,math.log10(changeHA(loss.data[0])))
  coordx.append(t)

  coordy.append(loss.data[0])
  #coordy.append(math.log10(changeHA(loss.data[0])))
  
  model.zero_grad()
  loss.backward()

  for param in model.parameters():
    param.data -= learning_rate * param.grad.data


plt.plot(coordx,coordy)
plt.axis(ymin=0, ymax=0.5)
#plt.axis(ymin=0, ymax=15000)

#print(max(coordy),min(coordy))
#print(max(coordy) - min(coordy))
plt.show()
