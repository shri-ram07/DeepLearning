from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
import os


#We will convert the images to an array using pillow
dir_ = "Datasets/smiles_dataset/training_set/"
features_ = []
labels_ = []

for filenames in os.listdir(dir_):
    el_ = Image.open(dir_+filenames).convert('1')
    features_.append(np.array(el_.getdata()))
    if filenames[0:5]=="happy":
        labels_.append([1,0])
    elif filenames[0:3]=="sad":
        labels_.append([0,1])

features_ = np.array(features_)/255
labels_ = np.array(labels_)    #one hot encoded lables


#Model Selection
model = Sequential()
model.add(Dense(1024 , input_dim = 1024 , activation = 'relu'))
model.add(Dense(1024 , activation = 'relu'))
model.add(Dense(512 , activation = 'relu'))
model.add(Dense(2 , activation = 'sigmoid'))

opt_ = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt_, metrics=['accuracy'])
model.fit(features_, labels_, epochs=1000, batch_size=32, verbose=2)
scores = model.evaluate(features_, labels_, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
tes__ = []
#testing
"""tes_ = Image.open("Datasets/smiles_dataset/test_set/happy_test.png").convert('1')
tes__.append(tes_.getdata())
tes__ = np.array(tes__)/255

print(model.predict(tes__).round())"""


