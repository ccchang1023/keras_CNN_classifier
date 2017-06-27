import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KERAS_BACKEND']='tensorflow'
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential  #sequential NN models
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# X shape(60000, 28x28), y shape(10,10000,)
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#data pre-processing
X_train = X_train.reshape(-1,1,28,28)/255.0 #(sample,channel,w,h)
X_test = X_test.reshape(-1,1,28,28)/255.0
#y_train : 3 -> [0,0,1,0,0,0,0,0,0,0]
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

#Build NN
model = Sequential()

#Conv layer 1 output shape(32,28,28)
model.add(Conv2D(
	filters=32,
	kernel_size=5,
	strides=(1,1),
	padding = 'same',  #padding method
	input_shape = (1,28,28), #channel,height,weight
	data_format ='channels_first',
	))
model.add(Activation('relu'))

#Pooling layer 1 (max pooling) output shape (32,14,14)
model.add(MaxPooling2D(
	pool_size=(2,2),
	strides=(2,2),
	padding = 'same', #padding method
	data_format ='channels_first',
	))

#Conv layer 2 output shape(64,14,14)
model.add(Conv2D(64,5,strides=(1,1),padding = 'same',data_format ='channels_first'))
model.add(Activation('relu'))

#Pooling layer 2 (max pooling) output shape (64,7,7)
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',data_format ='channels_first'))

#Fully connected layer 1 input shape (64*7*7), output shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#Fully connected layer 2 input shape (1024), output shape(10)
model.add(Dense(10))
model.add(Activation('softmax'))

#Define optimizer
adam = Adam(lr=1e-4)

#Add metrics to get more result
model.compile(
	optimizer = adam,
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'],
	)

print('Training--------------')
model.fit(X_train,y_train,epochs=1,batch_size=32,)

print('Testing---------------')
loss, accuracy = model.evaluate(X_test,y_test)

print('\ntest lost:', loss)
print('\ntest accuracy:', accuracy)








