from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

data = load_iris()
x=data.data
y=data.target.reshape(-1,1)   #making it 2-dimensional data


#doing one hot encoding

encoder = OneHotEncoder()
y = encoder.fit_transform(y).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(20, input_dim=4 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(20, input_dim=20 , activation='relu'))
model.add(Dense(3, input_dim=20 , activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.005), metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=32 , verbose =2)

print("Accuracy : ",model.evaluate(x_test, y_test))
