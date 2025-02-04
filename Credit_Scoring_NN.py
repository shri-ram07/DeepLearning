from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam,SGD

data = pd.read_csv('Datasets/Datasets/credit_data.csv')


x = data[["income","age","loan"]]
y = np.array(data["default"]).reshape(-1,1)

#output classes = 0 and 1   so we perform one hot encoding

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
model = Sequential()
model.add(Dense(10,input_dim=3, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))   # for O/P

optimizer = SGD(lr=0.0005)
model.compile()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5000 , verbose=2)
res = model.evaluate(x_test, y_test)
print(res)