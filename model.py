# Author: Kelly Smith
# Udacity Self-Driving Car Nanodegree Program
# Project 3 - End-to-End Deep Learning for Driving Car around Test Track

import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation 
from keras.layers.normalization import BatchNormalization
import pickle, json
from sklearn.model_selection import train_test_split


def getModel():
	# Try a keras network...
	model = Sequential([
    	Conv2D(32, 3, 3, border_mode='valid', input_shape=(16, 32, 3), activation='relu'),
    	MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),
    	Dropout(0.50),
    	Conv2D(32, 3, 3, border_mode='valid', activation='relu'),
    	# Conv2D(32, 3, 3, border_mode='same', activation='relu'),
    	MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),
    	Flatten(),
    	Dense(1, input_shape=(16*32*3,)),
    ])
	return model

def getNVIDIAModel():
	# Try the NVIDIA model, as described in their paper:
	# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	# TODO
	pass


""" 
Compile and train the model here.
"""
def trainModel(model):	
	model.summary()
	model.compile(optimizer='adam',
	          loss='mean_squared_error')
	history = model.fit(X_train, Y_train, batch_size=nb_batchSize, nb_epoch=nb_epochs, verbose=1, validation_data=(X_val, Y_val))
	return history

"""
Saves the model
"""
def saveModel(model,filename):
	with open(filename, "w") as text_file:
		print(json.dumps(model.to_json()), file=text_file)

"""
Saves the weights
"""
def saveWeights(model,filename):
	model.save_weights(filename)

"""
Main executable when called standalone.
"""
if __name__ == '__main__':
	# Import training data.
	print("Loading the dataset...",end="")
	with open('car_simulator.p', 'rb') as f:
		data = pickle.load(f)
	print("Complete!")

	# Split the data into training and validation sets.
	# Used code from http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
	print("Splitting the data into training and validation sets...",end="")
	X_train, X_val, Y_train, Y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)
	print("Complete!")

	# ------- Building the network ------------- #
	print("Building network model...", end="")
	model = getModel()
	print("Complete!")

	# ------- Training the network ------------- # 
	print("Training network...")
	nb_batchSize = 128
	nb_epochs = 5
	trainModel(model)
	print("Training complete!")

	# ------- Saving the network & weights ------------- # 
	print("Saving the model and weights...",end="")
	saveModel(model,"model.json")
	saveWeights(model,"model.h5")
	print("Save Complete!")

