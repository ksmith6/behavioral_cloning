import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation 
import pickle
from sklearn.model_selection import train_test_split


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

def getModel():
	# Try a keras network...
	model = Sequential([
    	Conv2D(32, 3, 3, border_mode='valid', input_shape=(16, 32, 3), activation='relu'),
    	MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),
    	Dropout(0.50),
    	Conv2D(32, 3, 3, border_mode='valid', activation='relu'),
    	MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),
    	Flatten(),
    	Dense(1, input_shape=(16*32*3,)),
    ])
	return model

def trainModel(model):
	# TODO: Compile and train the model here.
	model.summary()
	model.compile(optimizer='adam',
	          loss='mean_squared_error')
	history = model.fit(X_train, Y_train, batch_size=nb_batchSize, nb_epoch=nb_epochs, verbose=1, validation_data=(X_val, Y_val))
	return history

"""
Saves the model
"""
def saveModel(model,filename):
	json_string = model.to_json()
	with open(filename, "w") as text_file:
		text_file.write(json_string)

"""
Saves the weights
"""
def saveWeights(model,filename):
	model.save_weights(filename)


# ------- Building the network ------------- #
print("Building network model...", end="")
model = getModel()
print("Complete!")

# ------- Training the network ------------- # 
print("Training network...")
nb_batchSize = 128
nb_epochs = 20
trainModel(model)
print("Training complete!")

# ------- Saving the network & weights ------------- # 
print("Saving the model and weights...",end="")
saveModel(model,"model.json")
saveWeights(model,"model.h5")
print("Save Complete!")

## Load VGG 
# from keras.applications.vgg16 import VGG16
# model = VGG16(input_tensor=input_tensor, include_top=False)
