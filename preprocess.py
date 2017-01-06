import cv2
import pickle
import numpy as np
# Adapted from http://stackoverflow.com/questions/24662571/python-import-csv-to-list
import csv

# Define the columns 
columns = {
	'cImg':0, 
	'lImg':1, 
	'rImg':2, 
	'steer':3, 
	'throttle':4, 
	'brake':5, 
	'speed':6}

# Define the angle factor offset for left/right camera steering.
# Currently value of 0.75, as recommended by Kunfeng Chen in Slack chat.
ANGLE_FACTOR = 0.75


LR_ADJUST = 0.30

# Define a constant for determining whether or not to include 
# a training example that has a zero steering angle.  This should tend to 
# discount the effects of 0 steering angles.  Value should be between [0 1].
ZERO_STEER_KEEP_PROB = 1.0

# Set a fixed seed for repeatability of tests (1:1 comparison).
np.random.seed(12345)

def readDrivingLog():
	print("Loading driving_log.csv...")
	with open('driving_log.csv', 'r') as f:
		reader = csv.reader(f)
		driving_log= list(reader)
		nTrainingExamples =len(driving_log)
		print("Data set contains %d rows" % (nTrainingExamples-1))
	# Ignore the top row (column labels)
	return driving_log[1:]

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

	# For each training example
	for i in range(nTrainingExamples):
		
		# Print progress 
		if (i % progressIncrement == 0):
			print("%d / %d" % (i, nTrainingExamples))

		# Get the steering angle
		steerCenter = float(driving_log[i][columns['steer']])

		# Get a random number between 0 and 1
		r = np.random.uniform(0,1)
		# If the steering angle is zero and a random number is less than some threshold...
		if (steerCenter != 0 or (steerCenter == 0 and r < ZERO_STEER_KEEP_PROB)):
			
			# Center image
			imgC = preprocess(driving_log[i][columns['cImg']], smallShape)
			images.append(imgC)
			labels.append(steerCenter)

			

			# Left Image			
			imgL = preprocess(driving_log[i][columns['lImg']].strip(), smallShape)
			steerLeft = steerCenter + LR_ADJUST# abs(steerCenter * ANGLE_FACTOR)
			images.append(imgL)
			labels.append(steerLeft)

			# Right image
			imgR = preprocess(driving_log[i][columns['rImg']].strip(), smallShape)
			steerRight = steerCenter - LR_ADJUST #abs(steerCenter * ANGLE_FACTOR)
			images.append(imgR)
			labels.append(steerRight)

			# Flipped Images

			# Flip center image left-right
			images.append(cv2.flip(imgC, 1))
			labels.append(steerCenter * -1)

			# Flip left Image			
			images.append(cv2.flip(imgL, 1))
			labels.append(steerLeft * -1)

			# Flip right image
			images.append(cv2.flip(imgR, 1))
			labels.append(steerRight * -1)
			

	
	# Convert to np arrays
	npImgs = np.asarray(images)

	# Save the data
	pickleData(npImgs, labels)

def execute():
	

	print("Processing data...")
	process(readDrivingLog(), (32,16))
	print("Complete!")

if __name__ == '__main__':
	execute()
