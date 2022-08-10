from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D


def model1(n):
	#step1 Initializing CNN
	classifier = Sequential()

	# step2 adding 1st Convolution layer and Pooling layer
	classifier.add(Convolution2D(32,(3,3),input_shape = (100,100,3), activation = 'relu'))

	classifier.add(MaxPooling2D(pool_size = (2, 2)))

	# step3 adding 2nd convolution layer and polling layer
	classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

	classifier.add(MaxPooling2D(pool_size = (2, 2)))


	#step4 Flattening the layers
	classifier.add(Flatten())

	#step5 Full_Connection

	classifier.add(Dense(units=32,activation = 'relu'))

	classifier.add(Dense(units=64,activation = 'relu'))

	classifier.add(Dense(units=128,activation = 'relu'))

	classifier.add(Dense(units=256,activation = 'relu'))

	classifier.add(Dense(units=256,activation = 'relu'))

	classifier.add(Dense(units=n,activation = 'softmax'))

	#step6 Compiling CNN
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	return classifier

