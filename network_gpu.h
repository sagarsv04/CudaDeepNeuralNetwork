/*
 *  network.hpp
 *  Contains declaration of Network algorithm
 *
 *  Author :
 *  Sagar Vishwakarma (svishwa2@binghamton.edu)
 *  State University of New York, Binghamton
 */

#ifndef _NET_GPU_HPP_
#define _NET_GPU_HPP_

#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cstring>
#include <cmath>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace std;

#define BLOCK_SIZE 16

#define NET_DEBUG 0

#define SMALL_DATA_SIZE 0
#define DATA_SIZE 7998

#define SEED 42

class MyNN {

	public:
		int in_dimension;
		int op_dimension;
		int num_epochs;
		int num_neurons;
		float learning_rate;
		float train_accuracy;
		float test_accuracy;
		vector<float> loss;

		// data and labels
		vector<vector<float>> X_data;
		vector<vector<float>> Y_labels;

		// weight between input and first hidden layer
		vector<vector<float>> weight_one;
		vector<float> bias_one;
		vector<vector<float>> forward_one;

		// weight between first hidden layer and second hidden layer
		vector<vector<float>> weight_two;
		vector<float> bias_two;
		vector<vector<float>> forward_two;

		// weight between second hidden layer and output layer
		vector<vector<float>> weight_three;
		vector<float> bias_three;
		vector<vector<float>> forward_three;

		void set_parameters(int in_dimension, int op_dimension, int num_epochs, int num_neurons, float learning_rate);
		void initialize_neural_net(float low, float high);

		void feed_forward(void);
		void backprop(void);
		vector<int> predict(vector<vector<float>> &data);
		float error(void);

		void sigmoid(vector<vector<float>> &z, int forward_num);
		void softmax(vector<vector<float>> &z);
		vector<vector<float>> sigmoid_derv(vector<vector<float>> &forward_data);
		vector<vector<float>> cross_entropy(void);

		vector<vector<float>> transpose_vectors(vector<vector<float>> &vector_val);
		vector<float> sum_vector(vector<vector<float>> &forward_delta);
		void sum_bias_and_vectors(vector<vector<float>> &vector_val, int bias_num);
		vector<vector<float>> dot_product_vectors(vector<vector<float>> &vector_one, vector<vector<float>> &vector_two);
		vector<vector<float>> multiply_vectors(vector<vector<float>> &vector_one, vector<vector<float>> &vector_two);
		void update_weight_bias(vector<vector<float>> &forward_delta, int weight_bias_num);

		void print_weight_bias(bool all);
		void print_vectors(vector<float> vec);

		void train_model(void);
		void calculate_accuracy(vector<vector<float>> &data, vector<vector<float>> &labels, bool is_train);
		// MyNN();
	// ~MyNN();
};

vector<int> load_labels(string path);
vector<vector<float>> load_data(string path, bool normalize);
vector<vector<float>> create_one_hot_labels(vector<int> labels);

void print_one_hot_and_labels(vector<int> labels, vector<vector<float>> one_hot_labels, bool all);
void print_image_data(vector<vector<float>> data, vector<int> labels, bool all);


#endif /* _NET_GPU_HPP_ */
