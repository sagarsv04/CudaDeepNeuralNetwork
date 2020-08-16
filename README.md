# MNIST handwriting digit recognition

Link: https://www.cs.binghamton.edu/~kchiu/cs547/prog/4/


## Author :

Sagar Vishwakarma (svishwa2@binghamton.edu)

State University of New York, Binghamton


## File :

1)	Makefile                    - Compile the program
2)	mnist_dnn.cpp               - Contains cpp implementation of algorithm
3)	network.cpp                 - Contains implementation of MNIST handwriting recognize algorithm
4)	network.hpp                 - Contains declaration of MNIST handwriting recognize algorithm
5)	mnist_dnn.py                - Contains python implementation of algorithm


## Run :

- Open a terminal in project directory      : make (to build project)
- To run C++ cpu algorithm                 : ./mnist_dnn
- To run C++ gpu algorithm                 : ./mnist_dnn_gpu
- To run python mnist algorithm             : python mnist.py


## Report :

- I ran the code for 150 epochs:
- File : mnist_dnn_loss.png shows the loss of neural network with each epoch along with train and test accuracy (the plot was created using python and matplotlib).


- I have implemented a simple neural network with two hidden layers
- Number of neurons in hidden layers are 128
- The input layer consists 784 nodes that is 28x28 pixels of an image
- The input data is normalized between 0.0 to 1.0 based on pixel values
- The output layer consists 10 nodes that is 10 classes of images between 0 and 9
- One Hot Vector of size 10 for each label values
- I am using sigmoid activation function between layers, and softmax activation function for the output layer
- I am using cross_entropy and sigmoid_derivative to update weights and biases of layers in neural network


## Note :
- Use NET_DEBUG to print each operations info
- Use SMALL_DATA_SIZE and DATA_SIZE to work on small data size
