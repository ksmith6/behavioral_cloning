import cv2
import pickle
import numpy as np
# Adapted from http://stackoverflow.com/questions/24662571/python-import-csv-to-list
import csv

def readDrivingLog():
	print("Loading driving_log.csv...")
	with open('driving_log.csv', 'r') as f:
		reader = csv.reader(f)
		driving_log= list(reader)
		nTrainingExamples =len(driving_log)
		print("Data set contains %d rows" % nTrainingExamples)
	return driving_log

# Define the columns 
columns = {
	'cImg':0, 
	'lImg':1, 
	'rImg':2, 
	'steer':3, 
	'throttle':4, 
	'brake':5, 
	'speed':6}

""" 
Normalizes the image data from -0.5 to +0.5
"""
def normalize(X):
	# Normalize the image data to [xLow, xHigh]
	X = X.astype('float32')
	X = X/255.0 - 0.5
	return X

"""
Pre-processor: loads iamges, rescales, and normalizes domain from -0.5 to 0.5
"""
def preprocess(filename, smallShape):
	img = cv2.imread(filename)
	return preprocessImg(img,smallShape)

def preprocessImg(img, smallShape):
	# Re-scale the image
	img = cv2.resize(img,smallShape)
	
	# Disabled conversion to grayscale
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = normalize(img)
	return img


"""
Pickles the data.
"""
def pickleData(features, labels):
	with open('car_simulator.p','wb') as f:
		pickle.dump({'features':features, 'labels':labels},f)

"""
Master function for converting JPGs to features & labels.
"""
def process(driving_log, smallShape):
	progressIncrement = 100
	nTrainingExamples = len(driving_log)
	images = []
	labels = []
	for i in range(nTrainingExamples):
		
		# Print progress 
		if (i % progressIncrement == 0):
			print("%d / %d" % (i, nTrainingExamples))

		img = preprocess(driving_log[i][columns['cImg']], smallShape)
		
		images.append(img)
		labels.append(driving_log[i][columns['steer']])
	
	# Convert to np arrays
	npImgs = np.asarray(images)

	# Save the data
	pickleData(npImgs, labels)

def execute():
	print("Processing data...")
	process(readDrivingLog(), (32,16))
	print("Complete!")

execute()