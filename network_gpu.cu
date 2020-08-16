/*
 *  network.cpp
 *  Contains implementation of Network algorithm
 *
 *  Author :
 *  Sagar Vishwakarma (svishwa2@binghamton.edu)
 *  State University of New York, Binghamton
 */

#include "network_gpu.h"


auto reverse_int = [](int num) {
	unsigned char c1, c2, c3, c4;
	c1 = num & 255, c2 = (num >> 8) & 255, c3 = (num >> 16) & 255, c4 = (num >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};


__global__ void mat_mul_gpu(float* vec_one, float* vec_two, float* ret_vec, int vec_one_row, int vec_one_col, int vec_two_col) {
	// compute global thread coordinates
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// linearize coordinates for data access
	int offset = row * vec_two_col + col;
	// vec_one_col is equal to vec_two_row

	if ((row < vec_one_row) && (col < vec_two_col)) {
		float cum_sum = 0.0;
		for (int k = 0; k < vec_one_col; k++) {
			cum_sum += vec_one[row * vec_one_col + k] * vec_two[k * vec_two_col + col];
		}
		ret_vec[offset] = cum_sum;
	}
}


void print_one_hot_and_labels(vector<int> labels, vector<vector<float>> one_hot_labels, bool all) {

	int labels_size = labels.size();

	for (int i = 0; i < labels_size; i++) {
		printf("Label : %d  : ", labels[i]);
		printf("[");
		for (int j = 0; j < 10; j++) {
			printf(" %0.2f,", one_hot_labels[i][j]);
		}
		printf("]\n");

		if (all) {
			continue;
		}
		else {
			break;
		}
	}
}


vector<vector<float>> create_one_hot_labels(vector<int> labels) {

	int labels_size = labels.size();
	vector<vector<float>> one_hot_labels(labels_size, vector<float>(10, 0.0));

	for (int i = 0; i < labels_size; i++) {
		one_hot_labels[i][labels[i]] = 1.0;
	}
	return one_hot_labels;
}


vector<int> load_labels(string path) {

	ifstream file;
	vector<int> labels;

	int magic_number = 0;
	int num_labels = 0;

	file.open(path, ios::binary | ios::in);

	if (file.is_open()) {
		file.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		if(magic_number != 2049) {
			cerr << "Error Invalid Labes File" << endl;
		}
		else {
			file.read((char *) &num_labels, sizeof(num_labels));
			num_labels = reverse_int(num_labels);
			int label = 0;
			for(int i = 0; i < num_labels; i++) {
				file.read((char *) &label, 1);
				labels.push_back(label);
				if ((SMALL_DATA_SIZE)&&(i>DATA_SIZE)) {
					break;
				}
			}
			file.close();
		}
	}
	else {
		cerr << "Error Reading Labels at Path : " << path << endl;
		file.close();
	}
	return labels;
}


void print_image_data(vector<vector<float>> data, vector<int> labels, bool all) {

	int labels_size = labels.size();
	int image_size = data[0].size();

	for (int i = 0; i < labels_size; i++) {
		printf("Label : %d\n", labels[i]);
		// print 28*28 image
		for (int j = 0; j < image_size; j++) {
			if ((j%28)==0) {
				printf("\n");
			}
			if (data[i][j] > 0.0) {
				printf("*");
			}
			else{
				printf(" ");
			}
		}
		printf("\n");
		if (all) {
			continue;
		}
		else {
			break;
		}
	}
}


vector<vector<float>> load_data(string path, bool normalize) {

	ifstream file;
	vector<vector<float>> data;

	int magic_number = 0;
	int num_data = 0;
	int num_row = 0;
	int num_col = 0;

	file.open(path, ios::binary | ios::in);

	if (file.is_open()) {
		file.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverse_int(magic_number);
		if(magic_number != 2051) {
			cerr << "Error Invalid Data File" << endl;
		}
		else {
			file.read((char *) &num_data, sizeof(num_data));
			num_data = reverse_int(num_data);
			file.read((char *) &num_row, sizeof(num_row));
			num_row = reverse_int(num_row);
			file.read((char *) &num_col, sizeof(num_col));
			num_col = reverse_int(num_col);

			int image_size = num_row * num_col;
			vector<float> image(image_size, 0.0);
			int r_value = 0;
			float f_value = 0.0;
			for(int i = 0; i < num_data; i++) {
				for (int j = 0; j < image_size; j++) {
					file.read((char *) &r_value, 1);
					if (normalize) {
						f_value = r_value/255.0;
					}
					else {
						f_value = r_value;
					}
					image[j] = f_value;
				}
				data.push_back(image);
				if ((SMALL_DATA_SIZE)&&(i>DATA_SIZE)) {
					break;
				}
			}
			file.close();
		}
	}
	return data;
}


void MyNN::set_parameters(int in_dimension, int op_dimension, int num_epochs, int num_neurons, float learning_rate) {

	this->in_dimension = in_dimension;
	this->op_dimension = op_dimension;
	this->num_epochs = num_epochs;
	this->num_neurons = num_neurons;
	this->learning_rate = learning_rate;
	this->train_accuracy = 0.0;
	this->test_accuracy = 0.0;

}


void MyNN::initialize_neural_net(float low, float high) {

	default_random_engine generator;
	generator.seed(SEED);
	uniform_real_distribution<float> distr(low, high);

	this->weight_one.resize(this->in_dimension, vector<float>(this->num_neurons, 0.0));
	this->weight_two.resize(this->num_neurons, vector<float>(this->num_neurons, 0.0));
	this->weight_three.resize(this->num_neurons, vector<float>(this->op_dimension, 0.0));

	if (NET_DEBUG) {
		printf("Initializing Biases\n");
	}
	this->bias_one.resize(this->num_neurons, 0.0);
	this->bias_two.resize(this->num_neurons, 0.0);
	this->bias_three.resize(this->op_dimension, 0.0);

	if (NET_DEBUG) {
		printf("Initializing Weight One\n");
	}
	for (int i = 0; i < this->in_dimension; i++) {
		for (int j = 0; j < this->num_neurons; j++) {
			this->weight_one[i][j] = distr(generator);
		}
	}

	if (NET_DEBUG) {
		printf("Initializing Weight Two\n");
	}
	for (int i = 0; i < this->num_neurons; i++) {
		for (int j = 0; j < this->num_neurons; j++) {
			this->weight_two[i][j] = distr(generator);
		}
	}

	if (NET_DEBUG) {
		printf("Initializing Weight Three\n");
	}
	for (int i = 0; i < this->num_neurons; i++) {
		for (int j = 0; j < this->op_dimension; j++) {
			this->weight_three[i][j] = distr(generator);
		}
	}
}


void MyNN::print_weight_bias(bool all) {

	printf("Weight Bias One Values\n");
	for (int i = 0; i < this->in_dimension; i++) {
		printf("W-One %2d :: [", i);
		for (int j = 0; j < this->num_neurons; j++) {
			printf(" %0.2f,", this->weight_one[i][j]);
		}
		printf("]\n");
		if (all) {
			continue;
		}
		else {
			break;
		}
	}

	for (int j = 0; j < this->num_neurons; j++) {
		printf("B-One %2d :: %0.2f\n", j, this->bias_one[j]);
		if (all) {
			continue;
		}
		else {
			break;
		}
	}

	printf("Weight Bias Two Values\n");
	for (int i = 0; i < this->num_neurons; i++) {
		printf("W-Two %2d :: [", i);
		for (int j = 0; j < this->num_neurons; j++) {
			printf(" %0.2f,", this->weight_two[i][j]);
		}
		printf("]\n");
		if (all) {
			continue;
		}
		else {
			break;
		}
	}

	for (int j = 0; j < this->num_neurons; j++) {
		printf("B-Two %2d :: %0.2f\n", j, this->bias_two[j]);
		if (all) {
			continue;
		}
		else {
			break;
		}
	}

	printf("Weight Bias Three Values\n");

	for (int i = 0; i < this->num_neurons; i++) {
		printf("W-Three %2d :: [", i);
		for (int j = 0; j < this->op_dimension; j++) {
			printf(" %0.2f,", this->weight_three[i][j]);
		}
		printf("]\n");
		if (all) {
			continue;
		}
		else {
			break;
		}
	}

	for (int j = 0; j < this->op_dimension; j++) {
		printf("B-Three %2d :: %0.2f\n", j, this->bias_three[j]);
		if (all) {
			continue;
		}
		else {
			break;
		}
	}
}


void MyNN::print_vectors(vector<float> vec) {

	int vec_size = vec.size();
	printf("[ ");
	for (int i = 0; i < vec_size; i++) {
		printf(" %0.2f,", vec[i]);
	}
	printf("]\n");
}



void MyNN::sigmoid(vector<vector<float>> &z, int forward_num) {

	int z_row = z.size();
	int z_col = z[0].size();

	int f_row;
	int f_col;

	if (forward_num==1) {
		f_row = this->forward_one.size();
		if (f_row == 0) {
			this->forward_one.resize(z_row, vector<float>(z_col, 0.0));
			f_row = this->forward_one.size();
			f_col = this->forward_one[0].size();
		}
		else {
			f_col = this->forward_one[0].size();
		}
	}
	if (forward_num==2) {
		f_row = this->forward_two.size();
		if (f_row == 0) {
			this->forward_two.resize(z_row, vector<float>(z_col, 0.0));
			f_row = this->forward_two.size();
			f_col = this->forward_two[0].size();
		}
		else {
			f_col = this->forward_two[0].size();
		}
	}

	if (NET_DEBUG) {
		printf("Sigmoid %d Vector :: %d x %d <vs> Forward :: %d x %d\n", forward_num, z_row, z_col, f_row, f_col);
	}

	for (int r1 = 0; r1 < z_row; r1++) {
		for (int c1 = 0; c1 < z_col; c1++) {
			if (forward_num==1) {
				this->forward_one[r1][c1] = 1.0/(1.0 + exp(-z[r1][c1]));
			}
			if (forward_num==2) {
				this->forward_two[r1][c1] = 1.0/(1.0 + exp(-z[r1][c1]));
			}
		}
	}
}


void MyNN::softmax(vector<vector<float>> &z) {

	int z_row = z.size();
	int z_col = z[0].size();

	int f_col;
	int f_row = this->forward_three.size();

	if (f_row == 0) {
		this->forward_three.resize(z_row, vector<float>(z_col, 0.0));
		f_row = this->forward_three.size();
		f_col = this->forward_three[0].size();
	}
	else {
		f_col = this->forward_three[0].size();
	}

	if (NET_DEBUG) {
		printf("Softmax Vector :: %d x %d <vs> Forward :: %d x %d\n", z_row, z_col, f_row, f_col);
	}

	for (int r1 = 0; r1 < z_row; r1++) {
		float exp_sum = 0.0;
		for (int c1 = 0; c1 < z_col; c1++) {
			exp_sum += exp(z[r1][c1]);
		}
		for (int c1 = 0; c1 < z_col; c1++) {
			this->forward_three[r1][c1] = exp(z[r1][c1])/exp_sum;
		}
	}
}


vector<vector<float>> MyNN::sigmoid_derv(vector<vector<float>> &forward_data) {

	int forward_row = forward_data.size();
	int forward_col = forward_data[0].size();

	if (NET_DEBUG) {
		printf("Sigmoid Derv Forward :: %d x %d\n", forward_row, forward_col);
	}
	vector<vector<float>> ret_val(forward_row, vector<float>(forward_col, 0.0));;

	for (int r1 = 0; r1 < forward_row; r1++) {
		for (int c1 = 0; c1 < forward_col; c1++) {
			ret_val[r1][c1] = forward_data[r1][c1]*(1.0-forward_data[r1][c1]);
		}
	}

	return ret_val;
}


vector<vector<float>> MyNN::cross_entropy(void) {
	// between vector<vector<float>> forward_three, vector<vector<int>> Y_labels

	int forward_row = this->forward_three.size();
	int forward_col = this->forward_three[0].size();
	int label_row = this->Y_labels.size();
	int label_col = this->Y_labels[0].size();
	vector<vector<float>> ret_val;
	if ((forward_row != label_row) || (forward_col != label_col)) {
		printf("Error :: Invalid Vectors Size for Cross Entropy\n");
		printf("Forward :: %d x %d\n", forward_row, forward_col);
		printf("Labels :: %d x %d\n", label_row, label_col);
		return ret_val;
	}
	if (NET_DEBUG) {
		printf("Cross Entropy Labels :: %d x %d <vs> Forward :: %d x %d\n", label_row, label_col, forward_row, forward_col);
	}

	ret_val.resize(forward_row, vector<float>(forward_col, 0.0));

	for (int r1 = 0; r1 < forward_row; r1++) {
		float diff = 0.0;
		for (int c1 = 0; c1 < forward_col; c1++) {
			diff = this->forward_three[r1][c1] - this->Y_labels[r1][c1];
			ret_val[r1][c1] = diff/forward_row;
		}
	}

	return ret_val;
}


vector<vector<float>> MyNN::transpose_vectors(vector<vector<float>> &vector_val) {

	int vec_row = vector_val.size();
	int vec_col = vector_val[0].size();

	vector<vector<float>> ret_val(vec_col, vector<float>(vec_row, 0.0));

	for (int r1 = 0; r1 < vec_row; r1++) {
		for (int c1 = 0; c1 < vec_col; c1++) {
			ret_val[c1][r1] = vector_val[r1][c1];
		}
	}

	return ret_val;
}


vector<float> MyNN::sum_vector(vector<vector<float>> &forward_delta) {

	int forward_row = forward_delta.size();
	int forward_col = forward_delta[0].size();
	vector<float> ret_val(forward_col, 0.0);

	if (NET_DEBUG) {
		printf("Sum Vector :: %d x %d \n", forward_row, forward_col);
	}

	for (int c1 = 0; c1 < forward_col; c1++) {
		float col_sum = 0.0;
		for (int r1 = 0; r1 < forward_row; r1++) {
			col_sum += forward_delta[r1][c1];
		}
		ret_val[c1] = col_sum;
	}

	return ret_val;
}


void MyNN::sum_bias_and_vectors(vector<vector<float>> &vector_val, int bias_num) {

	int vec_row = vector_val.size();
	int vec_col = vector_val[0].size();
	int bias_row;

	if (bias_num==1) {
		bias_row = this->bias_one.size();
	}
	else if (bias_num==2) {
		bias_row = this->bias_two.size();
	}
	else if (bias_num==3) {
		bias_row = this->bias_three.size();
	}
	else {
		printf("Error :: Invalid Bias Num :: %d\n", bias_num);
		bias_row = 0;
	}

	if (vec_col != bias_row) {
		printf("Error :: Invalid Bias and Vector Size\n");
		printf("Vector :: %d x %d\n", vec_row, vec_col);
		printf("Bias :: %d x %d\n", 1, bias_row);
	}
	if (NET_DEBUG) {
		printf("Sum Vector :: %d x %d <vs> Bias %d :: %d x %d\n", vec_row, vec_col, bias_num, 1, bias_row);
	}

	for (int r1 = 0; r1 < vec_row; r1++) {
		for (int c1 = 0; c1 < vec_col; c1++) {
			if (bias_num==1) {
				vector_val[r1][c1] = vector_val[r1][c1] + this->bias_one[c1];
			}
			if (bias_num==2) {
				vector_val[r1][c1] = vector_val[r1][c1] + this->bias_two[c1];
			}
			if (bias_num==3) {
				vector_val[r1][c1] = vector_val[r1][c1] + this->bias_three[c1];
			}
		}
	}
}


vector<vector<float>> MyNN::dot_product_vectors(vector<vector<float>> &vector_one, vector<vector<float>> &vector_two) {

	int vec_one_row = vector_one.size();
	int vec_one_col = vector_one[0].size();
	int vec_two_row = vector_two.size();
	int vec_two_col = vector_two[0].size();
	vector<vector<float>> ret_val;
	if (vec_one_col != vec_two_row) {
		printf("Error :: Invalid Vectors Size for Dot Product\n");
		printf("Vector One :: %d x %d\n", vec_one_row, vec_one_col);
		printf("Vector Two :: %d x %d\n", vec_two_row, vec_two_col);
		return ret_val;
	}
	if (NET_DEBUG) {
		printf("Dot Product Vector :: %d x %d <vs> %d x %d\n", vec_one_row, vec_one_col, vec_two_row, vec_two_col);
	}
	ret_val.resize(vec_one_row, vector<float>(vec_two_col, 0.0));

	size_t vec_one_size = vec_one_row * vec_one_col * sizeof(float);
	size_t vec_two_size = vec_two_row * vec_two_col * sizeof(float);
	size_t ret_vec_size = vec_one_row * vec_two_col * sizeof(float);

	// allocate host memory (linear)
	float* h_vec_one = (float*)malloc(vec_one_size);
	float* h_vec_two = (float*)malloc(vec_two_size);
	float* h_ret_vec = (float*)malloc(ret_vec_size);

	// initialize host matrices
	int offset;
	for (int i = 0; i < vec_one_row; i++) {
		for (int j = 0; j < vec_one_col; j++) {
			offset = i * vec_one_col + j;
			h_vec_one[offset] = vector_one[i][j];
		}
	}
	for (int i = 0; i < vec_two_row; i++) {
		for (int j = 0; j < vec_two_col; j++) {
			offset = i * vec_two_col + j;
			h_vec_two[offset] = vector_two[i][j];
		}
	}

	// allocate device matrices
	float* d_vec_one;
	float* d_vec_two;
	float* d_ret_vec;
	cudaMalloc((void**)&d_vec_one, vec_one_size);
	cudaMalloc((void**)&d_vec_two, vec_two_size);
	cudaMalloc((void**)&d_ret_vec, ret_vec_size);

	// transfer to GPU
	cudaMemcpy(d_vec_one, h_vec_one, vec_one_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec_two, h_vec_two, vec_two_size, cudaMemcpyHostToDevice);
	if (NET_DEBUG) {
		printf("Copy to GPU done...\n");
	}

	// // kernel launch
	unsigned int grid_rows = (vec_one_row + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (vec_two_col + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	mat_mul_gpu<<<dimGrid, dimBlock>>>(d_vec_one, d_vec_two, d_ret_vec, vec_one_row, vec_one_col, vec_two_col);

	// wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// transfer to CPU
	cudaMemcpy(h_ret_vec, d_ret_vec, ret_vec_size, cudaMemcpyDeviceToHost);
	if (NET_DEBUG) {
		printf("Copy to CPU done...\n");
	}

	// copy back results to 2D Vectors
	for (int i=0; i < vec_one_row; i++) {
		for (int j=0; j < vec_two_col; j++) {
			offset = i * vec_two_col + j;
			ret_val[i][j] = h_ret_vec[offset];
		}
	}

	free(h_vec_one);
	free(h_vec_two);
	free(h_ret_vec);
	cudaFree(d_vec_one);
	cudaFree(d_vec_two);
	cudaFree(d_ret_vec);

	return ret_val;
}


vector<vector<float>> MyNN::multiply_vectors(vector<vector<float>> &vector_one, vector<vector<float>> &vector_two) {

	int vec_one_row = vector_one.size();
	int vec_one_col = vector_one[0].size();
	int vec_two_row = vector_two.size();
	int vec_two_col = vector_two[0].size();
	vector<vector<float>> ret_val;
	if ((vec_one_row != vec_two_row) || (vec_one_col != vec_two_col)) {
		printf("Error :: Invalid Vectors Size for Multiply\n");
		printf("Vector One :: %d x %d\n", vec_one_row, vec_one_col);
		printf("Vector Two :: %d x %d\n", vec_two_row, vec_two_col);
		return ret_val;
	}
	if (NET_DEBUG) {
		printf("Multiply Vector :: %d x %d <vs> %d x %d\n", vec_one_row, vec_one_col, vec_two_row, vec_two_col);
	}
	ret_val.resize(vec_one_row, vector<float>(vec_one_col, 0.0));

	for (int r1 = 0; r1 < vec_one_row; r1++) {
		for (int c1 = 0; c1 < vec_one_col; c1++) {
			ret_val[r1][c1] = vector_one[r1][c1]*vector_two[r1][c1];
		}
	}

	return ret_val;
}


float MyNN::error(void) {

	float error = -1.0;
	// error between forward_three and Y_data;
	int actual_rows = this->Y_labels.size();
	int actual_cols = this->Y_labels[0].size();
	int pred_rows = this->forward_three.size();
	int pred_cols = this->forward_three[0].size();

	if ((actual_rows != pred_rows) || (actual_cols != pred_cols)) {
		printf("Error :: Invalid Label and Prediction Size\n");
		printf("Actual Lable :: %d x %d\n", actual_rows, actual_cols);
		printf("Predicted Lable :: %d x %d\n", pred_rows, pred_cols);
		return error;
	}

	float log_p_sum = 0.0;

	for (int r1 = 0; r1 < pred_rows; r1++) {
		int cal_idx = -1;
		for (int c1 = 0; c1 < pred_cols; c1++) {
			if (this->Y_labels[r1][c1]==1.0) {
				cal_idx = c1;
			}
		}
		log_p_sum += -log(this->forward_three[r1][cal_idx]);
	}
	error = log_p_sum/actual_rows;

	return error;
}


void MyNN::update_weight_bias(vector<vector<float>> &forward_delta, int weight_bias_num) {

	int weight_row;
	int weight_col;
	int bias_row;

	if (NET_DEBUG) {
		printf("Updating Weights Bias for : %d\n", weight_bias_num);
	}

	if (weight_bias_num==3) {
		vector<vector<float>> forward_data_T = this->transpose_vectors(this->forward_two);
		vector<vector<float>> weight_update = this->dot_product_vectors(forward_data_T, forward_delta);
		weight_row = weight_update.size();
		weight_col = weight_update[0].size();
		if (NET_DEBUG) {
			printf("Weight Update Size : %d x %d\n", weight_row, weight_col);
		}
		for (int r1 = 0; r1 < weight_row; r1++) {
			for (int c1 = 0; c1 < weight_col; c1++) {
				this->weight_three[r1][c1] -= this->learning_rate*weight_update[r1][c1];
			}
		}
		vector<float> bias_update = this->sum_vector(forward_delta);
		bias_row = bias_update.size();
		if (NET_DEBUG) {
			printf("Bias Update Size : %d\n", bias_row);
		}
		for (int b1 = 0; b1 < bias_row; b1++) {
			this->bias_three[b1] -= this->learning_rate*bias_update[b1];
		}
	}
	else if (weight_bias_num==2) {
		vector<vector<float>> forward_data_T  = this->transpose_vectors(this->forward_one);
		vector<vector<float>> weight_update = this->dot_product_vectors(forward_data_T, forward_delta);
		weight_row = weight_update.size();
		weight_col = weight_update[0].size();
		if (NET_DEBUG) {
			printf("Weight Update Size : %d x %d\n", weight_row, weight_col);
		}
		for (int r1 = 0; r1 < weight_row; r1++) {
			for (int c1 = 0; c1 < weight_col; c1++) {
				this->weight_two[r1][c1] -= this->learning_rate*weight_update[r1][c1];
			}
		}
		vector<float> bias_update = this->sum_vector(forward_delta);
		bias_row = bias_update.size();
		if (NET_DEBUG) {
			printf("Bias Update Size : %d\n", bias_row);
		}
		for (int b1 = 0; b1 < bias_row; b1++) {
			this->bias_two[b1] -= this->learning_rate*bias_update[b1];
		}
	}
	else if (weight_bias_num==1) {
		vector<vector<float>> forward_data_T = this->transpose_vectors(this->X_data);
		vector<vector<float>> weight_update = this->dot_product_vectors(forward_data_T, forward_delta);
		weight_row = weight_update.size();
		weight_col = weight_update[0].size();
		if (NET_DEBUG) {
			printf("Weight Update Size : %d x %d\n", weight_row, weight_col);
		}
		for (int r1 = 0; r1 < weight_row; r1++) {
			for (int c1 = 0; c1 < weight_col; c1++) {
				this->weight_one[r1][c1] -= this->learning_rate*weight_update[r1][c1];
			}
		}
		vector<float> bias_update = this->sum_vector(forward_delta);
		bias_row = bias_update.size();
		if (NET_DEBUG) {
			printf("Bias Update Size : %d\n", bias_row);
		}
		for (int b1 = 0; b1 < bias_row; b1++) {
			this->bias_one[b1] -= this->learning_rate*bias_update[b1];
		}
	}
	else {
		printf("Error :: Failed to Update Weights and Biases for : %d\n", weight_bias_num);
	}
}


void MyNN::feed_forward(void) {

	vector<vector<float>> z_1 = this->dot_product_vectors(this->X_data, this->weight_one);
	this->sum_bias_and_vectors(z_1, 1);
	this->sigmoid(z_1, 1);
	z_1.clear();

	vector<vector<float>> z_2 = this->dot_product_vectors(this->forward_one, this->weight_two);
	this->sum_bias_and_vectors(z_2, 2);
	this->sigmoid(z_2, 2);
	z_2.clear();

	vector<vector<float>> z_3 = this->dot_product_vectors(this->forward_two, this->weight_three);
	this->sum_bias_and_vectors(z_3, 3);
	this->softmax(z_3);
	z_3.clear();
}


void MyNN::backprop(void) {

	float error = this->error();
	this->loss.push_back(error);
	printf("Loss calculated is :: %0.4f\n", error);

	vector<vector<float>> forward_3_delta = this->cross_entropy();
	this->update_weight_bias(forward_3_delta, 3);
	vector<vector<float>> weight_3_T = this->transpose_vectors(this->weight_three);
	vector<vector<float>> z_2_delta = this->dot_product_vectors(forward_3_delta, weight_3_T);
	forward_3_delta.clear();
	weight_3_T.clear();

	vector<vector<float>> forward_2_derv = this->sigmoid_derv(this->forward_two);
	vector<vector<float>> forward_2_delta = this->multiply_vectors(z_2_delta, forward_2_derv);
	this->update_weight_bias(forward_2_delta, 2);
	z_2_delta.clear();
	forward_2_derv.clear();

	vector<vector<float>> weight_2_T = this->transpose_vectors(this->weight_two);
	vector<vector<float>> z_1_delta = this->dot_product_vectors(forward_2_delta, weight_2_T);
	forward_2_delta.clear();
	weight_2_T.clear();

	vector<vector<float>> forward_1_derv = this->sigmoid_derv(this->forward_one);
	vector<vector<float>> forward_1_delta = this->multiply_vectors(z_1_delta, forward_1_derv);
	this->update_weight_bias(forward_1_delta, 1);
	z_1_delta.clear();
	forward_1_derv.clear();
	forward_1_delta.clear();

}


vector<int> MyNN::predict(vector<vector<float>> &data) {

	this->X_data = data;
	this->feed_forward();
	int f_row = this->forward_three.size();
	vector<int> pridict_labels(f_row, -1);
	float predic_col;
	float predic_val;
	float predic_pre_val;
	for (int r1 = 0; r1 < f_row; r1++) {
		predic_col = -1.0;
		predic_pre_val = -1.0;
		for (int c1 = 0; c1 < this->op_dimension; c1++) {
			predic_val = this->forward_three[r1][c1];
			if (predic_val>predic_pre_val) {
				predic_pre_val = predic_val;
				predic_col = (float)c1;
			}
		}
		pridict_labels[r1] = (int)predic_col;
	}
	this->X_data.clear();
	return pridict_labels;
}


void MyNN::calculate_accuracy(vector<vector<float>> &data, vector<vector<float>> &labels, bool is_train) {

	if (is_train) {
		printf("\nCalculating Train Accuracy ...\n");
	}
	else {
		printf("\nCalculating Test Accuracy ...\n");
	}

	float accuracy = 0.0;
	vector<int> pridict_labels = this->predict(data);
	int p_row = pridict_labels.size();
	int a_row = labels.size();
	int a_col = labels[0].size();

	if (a_row != p_row) {
		printf("Error :: Invalid Prediction Count\n");
		printf("Actual Count :: %d x %d\n", a_row, a_col);
		printf("Predicted Count :: %d x %d\n", p_row, 1);
	}
	if (NET_DEBUG) {
		printf("Calculated Prediction for %d Data\n", p_row);
	}

	int act_col;
	for (int p1 = 0; p1 < p_row; p1++) {
		act_col = -1;
		for (int c1 = 0; c1 < a_col; c1++) {
			if (labels[p1][c1]==1.0) {
				act_col = c1;
			}
		}
		if (act_col == pridict_labels[p1]) {
			accuracy += 1.0;
		}
		else {
			if (NET_DEBUG) {
				printf("Incorect Correct Prediction for %d data :: Actual %d  Predict %d\n", p1, act_col, pridict_labels[p1]);
			}
			// printf("Incorect Correct Prediction for %d data :: Actual %d  Predict %d\n", p1, act_col, pridict_labels[p1]);
		}
	}
	accuracy = (accuracy/p_row)*100;
	if (is_train) {
		printf("Train Accuracy :: %0.4f\n", accuracy);
		this->train_accuracy = accuracy;
	}
	else {
		printf("Test Accuracy :: %0.4f\n", accuracy);
		this->test_accuracy = accuracy;
	}
	pridict_labels.clear();
	this->X_data.clear();
	this->Y_labels.clear();
	this->forward_one.clear();
	this->forward_two.clear();
	this->forward_three.clear();
}


void MyNN::train_model(void) {

	printf("Training Model ...\n");
	for (int e = 0; e < this->num_epochs; e++) {
		printf("Epoch :: %02d\n", (e+1));
		this->feed_forward();
		this->backprop();
		printf("\n");
	}
	this->X_data.clear();
	this->Y_labels.clear();
	this->forward_one.clear();
	this->forward_two.clear();
	this->forward_three.clear();
}
