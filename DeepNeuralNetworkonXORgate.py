import numpy as np
from keras import Sequential
from keras.layers import Dense
import keras


X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

model = Sequential()  #Simple Sequential model
# we do not need to define number of input neurons in keras
# model.add(Dense(3, input_shape=(2,) , activation='relu'))
#we do not need to specify the input layer , we can directly put it to input_dim of the hidden layer
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=16,activation='relu')) #hidden layers
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#in above code we are adding nodes in the hidden layer in which units means the number
#of hidden neurons

#now we have to choose , how to model is going to find the
#minimum loss function , here we use SGD(Stochastic Gradient Descent) with leaning rate 0.3

#sgd = keras.optimizers.SGD(learning_rate=0.3)
#now we will compile this SGD in our model
#we can use adam also to calculate minimum loss function -- more reliable
model.compile(loss='mean_squared_error', optimizer="adam", metrics=['binary_accuracy'])

#fit the model
model.fit(X,Y ,  epochs=10000 , verbose=2)  #verbose 0 is silent , 1 and 2 are showing results
#batch_size means number of training samples and epochs is number of training iteration
print("Prediction is : ")
print(model.predict(X).round())

