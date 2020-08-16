# Make file for building application

CC = g++
CUDA = nvcc
CFLAGS = -Wall -Wextra -std=c++11 -lpthread -pedantic -O3 -ldl -pthread # for linux
# CFLAGS = -Wall -Wextra -std=c++11 -lpthread -pedantic -O3 -pthread # for windows MinGW
CUDAFLAGS = -std=c++11 -arch=sm_75


all: mnist_dnn mnist_dnn_gpu
	$(info Make sure you are using correct CFLAGS for OS)
#
cpu: mnist_dnn
	$(info Make sure you are using correct CFLAGS for OS)

gpu: mnist_dnn_gpu
	$(info Make sure you are using correct CFLAGS for OS)

mnist_dnn: mnist_dnn.o network.o
	$(info Building CPU Executable)
	$(CC) mnist_dnn.o network.o $(CFLAGS) -o mnist_dnn

mnist_dnn.o: mnist_dnn.cpp network.hpp
	$(CC) -c mnist_dnn.cpp network.hpp $(CFLAGS)

network.o: network.cpp network.hpp
	$(CC) -c network.cpp network.hpp $(CFLAGS)

# GPU Make Section

mnist_dnn_gpu: mnist_dnn_gpu.o network_gpu.o
	$(info Building GPU Executable)
	$(CUDA) mnist_dnn_gpu.o network_gpu.o -o mnist_dnn_gpu

mnist_dnn_gpu.o: mnist_dnn_gpu.cu network_gpu.h
	$(CUDA) -c mnist_dnn_gpu.cu $(CUDAFLAGS)

network_gpu.o: network_gpu.cu network_gpu.h
	$(CUDA) -c network_gpu.cu $(CUDAFLAGS)

clean:
	rm -f *.o *.d *.gch mnist_dnn mnist_dnn_gpu
