/*
 *  mnist_dnn.cpp
 *
 *  Author :
 *  Sagar Vishwakarma (svishwa2@binghamton.edu)
 *  State University of New York, Binghamton
 */

#include "network_gpu.h"



void train_test_model(int num_epochs, float learning_rate, string data_path) {

	int data_row;
	int data_col;
	int op_row;
	int op_col;

	vector<int> train_labels = load_labels(data_path + "train-labels-idx1-ubyte");
	vector<vector<float>> train_data = load_data(data_path + "train-images-idx3-ubyte", true);
	vector<vector<float>> one_hot_labels = create_one_hot_labels(train_labels);
	// print_one_hot_and_labels(train_labels, one_hot_labels, false);
	// print_image_data(train_data, train_labels, false);
	train_labels.clear();

	data_col = train_data[0].size();
	op_col = one_hot_labels[0].size();

	if (NET_DEBUG) {
		data_row = train_data.size();
		op_row = one_hot_labels.size();
		printf("Training Data Info ...\n");
		printf("Data Size :: %d x %d\n", data_row, data_col);
		printf("Out Size :: %d x %d\n", op_row, op_col);
	}

	MyNN model = MyNN();
	model.set_parameters(data_col, op_col, num_epochs, 128, learning_rate);
	model.initialize_neural_net(-0.5, 0.5);
	// model.print_weight_bias(false);
	model.X_data = train_data;
	model.Y_labels = one_hot_labels;
	model.train_model();

	model.calculate_accuracy(train_data, one_hot_labels, true);
	train_data.clear();
	one_hot_labels.clear();

	vector<int> test_labels = load_labels(data_path + "t10k-labels-idx1-ubyte");
	vector<vector<float>> test_data = load_data(data_path + "t10k-images-idx3-ubyte", true);
	one_hot_labels = create_one_hot_labels(test_labels);

	if (NET_DEBUG) {
		data_row = test_data.size();
		data_col = test_data[0].size();
		op_row = one_hot_labels.size();
		op_col = one_hot_labels[0].size();
		printf("Testing Data Info ...\n");
		printf("Data Size :: %d x %d\n", data_row, data_col);
		printf("Out Size :: %d x %d\n", op_row, op_col);
	}

	model.calculate_accuracy(test_data, one_hot_labels, false);
	test_data.clear();
	one_hot_labels.clear();
}


int main(int argc, char const* argv[]) {

	if (argc != 4) {
		cerr << "Help : Usage "<< argv[0] << " num_epochs learning_rate data_path" << endl;
		cerr << "Example : "<< argv[0] << " 150 0.2 ./data/" << endl;
		exit(1);
	}
	else {

		int num_epochs;
		float learning_rate;
		string data_path = argv[3];
		// assign argv values to variables
		num_epochs	= atoi(argv[1]);
		learning_rate = atof(argv[2]);
		if ((num_epochs<=0)||(learning_rate<=0.0)) {
			cerr << "Please Provide Valid Arguments"<< endl;
			cerr << "Example : "<< argv[0] << " 150 0.2 ./data/" << endl;
			exit(1);
		}
		else {
			// code goes here
			printf("num_epochs      :: %d\n", num_epochs);
			printf("learning_rate :: %f\n", learning_rate);
			printf("data_path     :: %s\n", data_path.c_str());

			train_test_model(num_epochs, learning_rate, data_path);
		}
	}
	return 0;
}
