import pyanitools as pya
import torch
import math
from torch.autograd import Variable

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




x = Variable(torch.FloatTensor(data))
y = Variable(torch.FloatTensor(output),requires_grad=False)



model = torch.nn.Sequential(
          torch.nn.Linear(3,1),
        )

"""
model = torch.nn.Sequential(
          torch.nn.Linear(2721,1),
          torch.nn.ReLU(),
          torch.nn.Linear(1,2721),
        )
"""

loss_fn = torch.nn.MSELoss(size_average=False)


learning_rate = 1e-4
for t in range(50000):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)

  #print(y_pred.size())
  print(t, loss.data[0])
  
  model.zero_grad()
  loss.backward()

  for param in model.parameters():
    param.data -= learning_rate * param.grad.data
