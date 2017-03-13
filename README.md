# dl_framework
dl_framework

A library that demonstrates training of data using stochastic gradient descent method

The minflow library is a part of Udacity's Nanodegree Program and has been prepared while pursuing the same

Table of Contents

dl_framework.py

This file consists of the set of functions used to perform a basic back propogation in a neural network

nn_test.ipynb

This file consists of a sample neural network test

boston_housing_with_custom_dl_framework.ipynb

This file consists predetion of Boston Housing Data using our dl_framework

Usage

Include the dl_framework.py in your root project and use the following classess as follows:

Input()

Use it to declare input nodes of neural network

Linear()

Use to declare a node performing the task of linear activation of form Y = XW+ b

Sigmoid()

Use to declare a node performing sigmodial activation

MSE

Use this node to calculate Mean Square Error

Further functions are included to perform forward pass, backward pass and gradient descent
In boston_housing_with_custom_dl_framework.ipynb  sci-kit learn library is used to resample the dataset to perform a Stochastic Gradient Descent. 
The dl_framework library is designed for Stochastic Gradient Descent and hence the function for gradient descent is named as sgd_update
