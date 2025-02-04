from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

data = load_iris()
x=data.data
y=data.target.reshape(-1,1)   #making it 2-dimensional data


