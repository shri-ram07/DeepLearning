from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense , Dropout , BatchNormalization
from keras.optimizers import SGD
(train_x , y_train),(test_x , y_test) = cifar10.load_data()


#doing one hot encoding in labels
y_train = to_categorical(y_train )
y_test = to_categorical(y_test )


#Normalization of features
train_x = train_x/255.0
test_x = test_x/255.0



#Model Building
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3),padding='same',activation='relu',input_shape=(32,32,3) , kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))




#Model Training
opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_x, y_train, batch_size=64, epochs=50 , validation_data=(test_x, y_test) , verbose=1)
score = model.evaluate(test_x, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

