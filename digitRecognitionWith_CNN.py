from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator



(X_train , y_train),(X_test , y_test) = mnist.load_data()
"""plt.imshow(X_train[50000],cmap='gray')
plt.show()"""

#tensorflow can handle image for mat in : (batch,height,width,channel)
features_train = X_train.reshape(60000,28,28,1)
features_test = X_test.reshape(10000,28,28,1)


#Normalization
features_train = features_train.astype('float32')
features_test = features_test.astype('float32')
features_train /= 255
features_test /= 255


#One hot encoding
target_train = np_utils.to_categorical(y_train,10)
target_test = np_utils.to_categorical(y_test,10)


#Model building
model = Sequential()

model.add(Conv2D(32 , kernel_size=(3,3), input_shape=(28,28,1)  ,strides = (1,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32 , (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64 , 3))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64 , 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
"""model.fit(features_train, target_train, batch_size=32, epochs=15, validation_data=(features_test, target_test) , verbose=1)
score = model.evaluate(features_test, target_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])"""
"""
1875/1875 [==============================] - 52s 7ms/step - loss: 0.1047 - accuracy: 0.9679 - val_loss: 0.0279 - val_accuracy: 0.9902
Epoch 2/5
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0502 - accuracy: 0.9845 - val_loss: 0.0359 - val_accuracy: 0.9889
Epoch 3/5
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0417 - accuracy: 0.9880 - val_loss: 0.0245 - val_accuracy: 0.9924
Epoch 4/5
1875/1875 [==============================] - 12s 7ms/step - loss: 0.0347 - accuracy: 0.9894 - val_loss: 0.0244 - val_accuracy: 0.9930
Epoch 5/5
1875/1875 [==============================] - 14s 7ms/step - loss: 0.0291 - accuracy: 0.9909 - val_loss: 0.0247 - val_accuracy: 0.9923
313/313 [==============================] - 1s 4ms/step - loss: 0.0247 - accuracy: 0.9923
Test loss: 0.024737324565649033
Test accuracy: 0.9922999739646912
"""
#model.fit(features_train, targets_train, batch_size=128, epochs=2, validation_data=(features_test,targets_test), verbose=1)

#score = model.evaluate(features_test, targets_test)
#print('Test accuracy: %.2f' % score[1])

#data augmentation helps to reduce overfitting
train_generator = ImageDataGenerator(rotation_range=7, width_shift_range=0.05, shear_range=0.2,
                         height_shift_range=0.07, zoom_range=0.05)

test_genrator = ImageDataGenerator()

train_generator = train_generator.flow(features_train, target_train, batch_size=64)
test_generator = test_genrator.flow(features_test, target_test, batch_size=64)

model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=5,
                    validation_data=test_generator, validation_steps=10000//64)

