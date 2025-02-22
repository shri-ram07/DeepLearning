import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM ,Dense ,Dropout
from optree.integration import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
NUM_OF_PREV_ITEM = 5


def reconstruct_data(data,num_prev=1):
    x , y = [],[]
    for i in range(len(data)-num_prev-1):
        a=data[i:i+num_prev,0]
        x.append(a)
        y.append(data[i+num_prev,0])

    return np.array(x),np.array(y)




#We want to result to be same each time
np.random.seed(1)

data = pd.read_csv("Datasets/Datasets/daily_min_temperatures.csv",usecols=[1])
"""plt.plot(data)
plt.show()"""
data = data.values
data=data.astype('float32')

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)


train , test = data[0:int(len(data)*0.7),:], data[int(len(data)*0.7):len(data),:]

#create the training data and test dataa metrix
train_x , train_y = reconstruct_data(train, NUM_OF_PREV_ITEM)
test_x , test_y = reconstruct_data(test, NUM_OF_PREV_ITEM)


#reshape input to be [numOFsamples , time steps , numOFfeatures]
train_x = np.reshape(train_x , (train_x.shape[0],1,train_x.shape[1]))
test_x = np.reshape(test_x , (test_x.shape[0],1,test_x.shape[1]))

model = Sequential()
model.add(LSTM(units=100,return_sequences=True,input_shape=(1,NUM_OF_PREV_ITEM)))
model.add(Dropout(0.5))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=100))
model.add(Dropout(0.3))
model.add(Dense(units=1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(train_x, train_y, epochs=10, batch_size=16,verbose=2)

test_predict = model.predict(test_x)
test_predict = scaler.inverse_transform(test_predict)
test_labels = scaler.inverse_transform([test_y])

test_score = mean_squared_error(test_labels[0], test_predict[:,0])
print("Score : %2f MSE"%test_score)

test_predict_plot = np.empty_like(data)
test_predict_plot[:,:]=np.nan
test_predict_plot[len(train_x)+2*NUM_OF_PREV_ITEM+1:len(data)-1] = test_predict
plt.plot(scaler.inverse_transform(data))
plt.plot(test_predict_plot,color='red')
plt.plot(test_labels,color='blue')
plt.show()


