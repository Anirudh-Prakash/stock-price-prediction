# stock-price-prediction
A Python program to predict the next 30 days stock prices based on the historic data.

## Requirements
* Numpy
* Pandas
* Tensorflow
* Keras
* Matplotlib

## LSTM for time series prediction
Recurrent neural networks (RNN) have proved one of the most powerful models for processing sequential data.

Long Short-Term memory is one of the most successful RNNs architectures. LSTM introduces the memory cell, a unit of computation that replaces traditional artificial neurons in the hidden layer of the network. With these memory cells, networks are able to effectively associate memories and input remote in time, hence suit to grasp the structure of data dynamically over time with high prediction capacity.

In this document, we have used the historical data (previous 5 years) of YESBANK.NS from Yahoo Finance and used it for training and validation process.

## Technical Indicators
Several Technical Indicators such as SMA ( Simple Moving average), EMA (Exponential Moving Average) etc. were derived for analysis of the dataset. The indicators were not used in training the model though they can be heavily used if we predict the price of the next day.

#### Note: This project was made for TechGig's Machine Learning Hackathon.
