import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pyanitools as pya
import math
import torch.nn.functional as F
import torch.nn.init as init

#FUNCTIONS TO PREPARE THE DATA
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
#FUNCTIONS TO TEST OTHER INPUT TRAINING
def cos():
	training_data = np.random.random_sample((500,1))
	output_data = []
	for i in range(len(training_data)):
		output_data.append(math.cos(training_data[i]))
	output_data = np.array(output_data)
	return training_data,output_data
def solo_aqui(x):
	x_train = []
	for i in range(len(x)):
		x_train.append(x[i][0])
	return x_train


hdf5file = 'gdb11_S01_06r.h5'
adl = pya.anidataloader(hdf5file)
coordinates,output_training = extract_mol(adl)
training_set = combinations(coordinates)
adl.cleanup()


#INPUT #6
training_set,output_training = cos()

#DATA SET
training_set,test_set,output_training,output_test = ten_percent(training_set,output_training)


feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]


#estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)	
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)	

x_train = np.array(solo_aqui(training_set))
y_train = output_training

x_eval = np.array(solo_aqui(test_set))
y_eval = output_test

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)


# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

predictions = list(estimator.predict(input_fn=eval_input_fn))

predicted_classes = [p["predictions"] for p in predictions]

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

final = []
for i in range(len(y_eval)):
	final.append(predicted_classes[i][0])

plt.plot(y_eval,y_eval)
plt.scatter(y_eval,np.array(final))

plt.show()
print(y_eval.shape,np.array(final).shape)
print(y_eval[10:15],"esta es la entrada")
print("						")
print(final[10:15])