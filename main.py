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
import os


#FUNCTIONS TO PREPARE THE DATA
def ten_percent(data,output):
	ten_percent = int(len(data) * 0.1)	
	training_set = data[ten_percent:]
	test_set  = data[:ten_percent]
	output_training = output[ten_percent:]
	output_test = output[:ten_percent]

	return np.array(training_set),np.array(test_set),output_training,output_test
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


def solo_aqui(x):
	x_train = []
	for i in range(len(x)):
		x_train.append(x[i][0])
	return np.float32(x_train)
def solo_aqui2(x):
	x_train = []
	for i in range(len(x)):
		x_train.append(x[i][1])
	return np.float32(x_train)
def solo_aqui3(x):
	x_train = []
	for i in range(len(x)):
		x_train.append(x[i][2])
	return np.float32(x_train)


#OTHER HYPER PARAMETERS
batch_size = 4
num_epochs = 1
steps = 1000


def main(estimator_,optimizer_,learning_rate,HU,activation_fn_):

	#hdf5file = 'ani_gdb_s01.h5'
	hdf5file = 'gdb11_S01_06r.h5'
	adl = pya.anidataloader(hdf5file)
	coordinates,output_training = extract_mol(adl)
	training_set = combinations(coordinates)
	adl.cleanup()

	#DATA SET

	training_set_shift = np.array(training_set).mean()
	training_set = training_set - training_set_shift
		
	
	training_set,test_set,output_training,output_test = ten_percent(training_set,output_training)


	tf.nn.l2_normalize(training_set, 0, epsilon=1e-12)
	tf.nn.l2_normalize(test_set, 0, epsilon=1e-12)

	feature_column_1 = tf.feature_column.numeric_column("x1")
	feature_column_2 = tf.feature_column.numeric_column("x2")
	feature_column_3 = tf.feature_column.numeric_column("x3")

	x_train_1 = solo_aqui(training_set)
	x_train_2 = solo_aqui2(training_set)
	x_train_3 = solo_aqui3(training_set)


	y_train = np.float32(output_training)

	x_eval_1 = solo_aqui(test_set)
	x_eval_2 = solo_aqui2(test_set)
	x_eval_3 = solo_aqui3(test_set)


	y_eval = np.float32(output_test)
	if estimator_ == tf.estimator.LinearRegressor:
		estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate))
	elif estimator_== tf.estimator.DNNRegressor:
		if optimizer_ == tf.train.GradientDescentOptimizer or optimizer_ == tf.train.AdagradOptimizer or optimizer_ == tf.train.AdadeltaOptimizer :
			estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate),hidden_units=HU)
		else:
			estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08),hidden_units=HU,activation_fn=activation_fn_)
	input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_train_1,"x2": x_train_2, "x3":x_train_3}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_train_1,"x2": x_train_2, "x3":x_train_3}, y_train, batch_size=batch_size, num_epochs=num_epochs, shuffle=False)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_eval_1,"x2": x_eval_2, "x3":x_eval_3}, y_eval, batch_size=batch_size	, num_epochs=num_epochs, shuffle=False)


	# We can invoke 1000 training steps by invoking the  method and passing the
	# training data set.
	estimator.train(input_fn=input_fn, steps=steps)

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

	print("							")
	print(y_eval[0:5],"esta es la entrada")
	print(final[0:5])
	print("							")
	print("estimator:",estimator_)
	print("optimizer:",optimizer_)
	print("learning_rate:",learning_rate)
	print("hidden_units:",HU)
	print("act function", activation_fn_)
	#plt.plot(y_eval,y_eval)
	plt.scatter(y_eval,np.array(final))

	plt.show()


def main_Adam(estimator_,optimizer_,learning_rate,HU,activation_fn_):

	#hdf5file = 'ani_gdb_s01.h5'
	hdf5file = 'gdb11_S01_06r.h5'
	adl = pya.anidataloader(hdf5file)
	coordinates,output_training = extract_mol(adl)
	training_set = combinations(coordinates)
	adl.cleanup()

	#DATA SET

	training_set_shift = np.array(training_set).mean()
	training_set = training_set - training_set_shift
		
	
	training_set,test_set,output_training,output_test = ten_percent(training_set,output_training)


	tf.nn.l2_normalize(training_set, 0, epsilon=1e-12)
	tf.nn.l2_normalize(test_set, 0, epsilon=1e-12)

	feature_column_1 = tf.feature_column.numeric_column("x1")
	feature_column_2 = tf.feature_column.numeric_column("x2")
	feature_column_3 = tf.feature_column.numeric_column("x3")

	x_train_1 = solo_aqui(training_set)
	x_train_2 = solo_aqui2(training_set)
	x_train_3 = solo_aqui3(training_set)


	y_train = np.float32(output_training)

	x_eval_1 = solo_aqui(test_set)
	x_eval_2 = solo_aqui2(test_set)
	x_eval_3 = solo_aqui3(test_set)


	y_eval = np.float32(output_test)
	if estimator_ == tf.estimator.LinearRegressor:
		estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08))
	elif estimator_== tf.estimator.DNNRegressor:
	
		estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08),hidden_units=HU,activation_fn=activation_fn_)
	input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_train_1,"x2": x_train_2, "x3":x_train_3}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_train_1,"x2": x_train_2, "x3":x_train_3}, y_train, batch_size=batch_size, num_epochs=num_epochs, shuffle=False)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_eval_1,"x2": x_eval_2, "x3":x_eval_3}, y_eval, batch_size=batch_size	, num_epochs=num_epochs, shuffle=False)


	# We can invoke 1000 training steps by invoking the  method and passing the
	# training data set.
	estimator.train(input_fn=input_fn, steps=steps)

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

	print("							")
	print(y_eval[0:5],"esta es la entrada")
	print(final[0:5])
	print("							")
	print("estimator:",estimator_)
	print("optimizer:",optimizer_)
	print("learning_rate:",learning_rate)
	print("hidden_units:",HU)
	print("act function", activation_fn_)
	#plt.plot(y_eval,y_eval)
	plt.scatter(y_eval,np.array(final))

	plt.show()
def main_RMS(estimator_,optimizer_,learning_rate,HU,activation_fn_):

	#hdf5file = 'ani_gdb_s01.h5'
	hdf5file = 'gdb11_S01_06r.h5'
	adl = pya.anidataloader(hdf5file)
	coordinates,output_training = extract_mol(adl)
	training_set = combinations(coordinates)
	adl.cleanup()

	#DATA SET

	training_set_shift = np.array(training_set).mean()
	training_set = training_set - training_set_shift
		
	training_set,test_set,output_training,output_test = ten_percent(training_set,output_training)


	tf.nn.l2_normalize(training_set, 0, epsilon=1e-12)
	tf.nn.l2_normalize(test_set, 0, epsilon=1e-12)

	feature_column_1 = tf.feature_column.numeric_column("x1")
	feature_column_2 = tf.feature_column.numeric_column("x2")
	feature_column_3 = tf.feature_column.numeric_column("x3")

	x_train_1 = solo_aqui(training_set)
	x_train_2 = solo_aqui2(training_set)
	x_train_3 = solo_aqui3(training_set)


	y_train = np.float32(output_training)

	x_eval_1 = solo_aqui(test_set)
	x_eval_2 = solo_aqui2(test_set)
	x_eval_3 = solo_aqui3(test_set)


	y_eval = np.float32(output_test)
	if estimator_ == tf.estimator.LinearRegressor:
		estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate,decay=0.9,momentum=0.0,epsilon=1e-10))
	elif estimator_== tf.estimator.DNNRegressor:
		estimator = estimator_(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=optimizer_(learning_rate,decay=0.9,momentum=0.0,epsilon=1e-10),hidden_units=HU)
	input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_train_1,"x2": x_train_2, "x3":x_train_3}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_train_1,"x2": x_train_2, "x3":x_train_3}, y_train, batch_size=batch_size, num_epochs=num_epochs, shuffle=False)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    {"x1": x_eval_1,"x2": x_eval_2, "x3":x_eval_3}, y_eval, batch_size=batch_size	, num_epochs=num_epochs, shuffle=False)


	# We can invoke 1000 training steps by invoking the  method and passing the
	# training data set.
	estimator.train(input_fn=input_fn, steps=steps)

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

	print("							")
	print(y_eval[0:5],"esta es la entrada")
	print(final[0:5])
	print("							")
	print("estimator:",estimator_)
	print("optimizer:",optimizer_)
	print("learning_rate:",learning_rate)
	print("hidden_units:",HU)
	print("act function", activation_fn_)
	#plt.plot(y_eval,y_eval)
	plt.scatter(y_eval,np.array(final))

	plt.show()




#estimator = tf.estimator.LinearRegressor(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=tf.train.GradientDescentOptimizer(0.0001))	
#estimator = tf.estimator.DNNRegressor(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=tf.train.GradientDescentOptimizer(0.01),hidden_units=[32])	
#estimator = tf.estimator.DNNRegressor(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-08),hidden_units=[32,32], activation_fn=tf.tanh)	
#estimator = tf.estimator.DNNRegressor(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00001,decay=0.9,momentum=0.0,epsilon=1e-10),hidden_units=[32,32], activation_fn=tf.tanh)	
#estimator = tf.estimator.DNNRegressor(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=tf.train.AdagradOptimizer(learning_rate=0.0001),hidden_units=[32,32], activation_fn=tf.tanh)	
#estimator = tf.estimator.DNNRegressor(feature_columns=[feature_column_1,feature_column_2,feature_column_3],optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.001),hidden_units=[32,32], activation_fn=tf.tanh)	

estimators = [tf.estimator.LinearRegressor,tf.estimator.DNNRegressor]
optimizers = [tf.train.GradientDescentOptimizer,tf.train.AdamOptimizer,tf.train.RMSPropOptimizer,tf.train.AdagradOptimizer,tf.train.AdadeltaOptimizer]
learning_rates = [0.01,0.001,0.0001,0.00001]
hidden_units = [[12,12,4]]
activation_fn = [tf.tanh,tf.sigmoid,tf.nn.elu,tf.nn.softplus,tf.nn.softsign,tf.nn.relu,tf.nn.relu6]


for i in estimators:
	for j in optimizers:
		for k in learning_rates:
			for l in hidden_units:
				for m in activation_fn :
					if j == tf.train.GradientDescentOptimizer or j == tf.train.AdagradOptimizer or j == tf.train.AdadeltaOptimizer:
						main(i,j,k,l,m)

					elif j == tf.train.AdamOptimizer:
						main_Adam(i,j,k,l,m)

					elif j == tf.train.RMSPropOptimizer:
						main_RMS(i,j,k,l,m)



#main(tf.estimator.DNNRegressor,tf.train.GradientDescentOptimizer,0.01,[12,12],tf.tanh)