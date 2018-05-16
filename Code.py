import numpy
from numpy import newaxis
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset_test(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)+1, 0]
        dataX.append(a)
    for i in range(len(dataset)-look_back-1):
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_train(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)+1, 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv('HDFCBANK.NS.csv', sep=",")
dataframe.dropna(inplace=True)
del dataframe['Date']
del dataframe['High']
del dataframe['Low']
del dataframe['Adj Close']
del dataframe['Open']
del dataframe['Volume']
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
look_back = 29
# split into train and test sets
train_size = int(len(dataset) * 0.915)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainX, trainY = create_dataset_train(train, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back+1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)

# make predictions
for i in range (1,40):
    testX, testY = create_dataset_test(test, look_back)
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testPredict = model.predict(testX)
    temp=testPredict[len(testPredict)-1,0]
    test=numpy.append(test,[[temp]],axis=0)
    test=numpy.delete(test,0,axis=0)
       
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
print (testPredict)
dataset=scaler.inverse_transform(dataset)
close_day0_actual=dataset[len(dataset)-1]
close_day30_predicted=testPredict[len(testPredict)-1]
return_predict=((close_day30_predicted-close_day0_actual)/close_day0_actual)*100
print(return_predict)

