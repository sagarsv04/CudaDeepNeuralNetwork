#/usr/bin/pthon
import os
import sys
import numpy as np
from tqdm import tqdm
import pickle
from mnist import MNIST
import argparse

np.random.seed(42)


MODEL_OVER_WRIGHT = False


def load_model(save_name):
	model = None
	if os.path.exists("./out/{0}.pkl".format(save_name)):
		with open("./out/{0}.pkl".format(save_name), "rb") as file:
			model = pickle.load(file)
	else:
		print("No File at ... {0}/out/{1}.pkl".format(os.getcwd(), save_name))
	return model


def load_data(train_data=True, normalize=False):

	mnist_data = MNIST("./data/")
	# mnist_data.gz = True
	data, labels = None, None

	if train_data:
		print("Loading Training Data ...")
		data, labels = mnist_data.load_training()
	else:
		print("Loading Testing Data ...")
		data, labels = mnist_data.load_testing()

	data = np.array(data)
	labels = np.array(labels)

	if normalize:
		data = data/data.max()

	return data, labels


# Activation
def sigmoid(s):
	return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
	return s * (1 - s)

def softmax(s):
	exps = np.exp(s - np.max(s, axis=1, keepdims=True))
	return exps/np.sum(exps, axis=1, keepdims=True)

# Loss Function
def cross_entropy(pred, real):
	n_samples = real.shape[0]
	res = pred - real
	return res/n_samples

# Error Function
def error(pred, real):
	# pred, real = a3, y
	n_samples = real.shape[0]
	logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
	loss = np.sum(logp)/n_samples
	return loss

# Create One Hot Vector Error Function
def get_one_hot_vector_array(labels):

	print("Creating One Hot Vector for Labels ...")
	one_hot_vector_size = np.unique(labels).shape[0]
	labels_one_hot_array = np.zeros((labels.shape[0], one_hot_vector_size))

	for idx in range(labels.shape[0]):
		# idx = 1
		labels_one_hot_array[idx][labels[idx]] = 1

	return labels_one_hot_array



class MyNN:
	def __init__(self, x, y, epoch):
		self.x = x
		neurons = 128
		self.lr = 0.5
		# x = train_data # (60000, 784)
		# y = train_labels # (60000, 10)
		ip_dim = x.shape[1]
		op_dim = y.shape[1]
		self.epoch = epoch
		self.train_accuracy = 0
		self.test_accuracy = 0
		self.loss = []
		# weight between input and first hidden layer
		self.w1 = np.random.randn(ip_dim, neurons)
		# w1.shape # (784, 128)
		self.b1 = np.zeros((1, neurons))
		# b1.shape # (1, 128)
		# weight between first hidden layer and second hidden layer
		self.w2 = np.random.randn(neurons, neurons)
		# w2.shape # (128, 128)
		self.b2 = np.zeros((1, neurons))
		# b2.shape # (1, 128)
		# weight between second hidden layer and output layer
		self.w3 = np.random.randn(neurons, op_dim)
		# w3.shape # (128, 10)
		self.b3 = np.zeros((1, op_dim))
		# b3.shape # (1, 10)
		self.y = y

	def feedforward(self):
		z1 = np.dot(self.x, self.w1) + self.b1
		# z1 = np.dot(x, w1) + b1
		# z1.shape # (60000, 128)
		self.a1 = sigmoid(z1)
		# a1.shape # (60000, 128)
		z2 = np.dot(self.a1, self.w2) + self.b2
		# z2 = np.dot(a1, w2) + b2
		# z2.shape # (60000, 128)
		self.a2 = sigmoid(z2)
		# a2.shape # (60000, 128)
		z3 = np.dot(self.a2, self.w3) + self.b3
		# z3 = np.dot(a2, w3) + b3
		# z3.shape # (60000, 10)
		self.a3 = softmax(z3)
		# a3.shape # (60000, 10)

	def backprop(self):
		loss = error(self.a3, self.y)
		self.loss.append(loss)
		a3_delta = cross_entropy(self.a3, self.y) # w3
		# a3_delta = cross_entropy(a3, y) # w3
		# a3_delta.shape # (60000, 10)
		z2_delta = np.dot(a3_delta, self.w3.T)
		# z2_delta = np.dot(a3_delta, w3.T)
		# z2_delta.shape # (60000, 128)
		a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
		# a2_delta = z2_delta * sigmoid_derv(a2) # w2
		# a2_delta.shape # (60000, 128) # element wise multiply
		z1_delta = np.dot(a2_delta, self.w2.T)
		# z1_delta.shape # (60000, 128)
		a1_delta = z1_delta * sigmoid_derv(self.a1) # w1
		# a1_delta.shape # (60000, 128)

		self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
		# np.dot(self.a2.T, a3_delta).shape # (128, 10)
		self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
		# np.sum(a3_delta, axis=0, keepdims=True).shape # (1, 10)
		self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
		self.b2 -= self.lr * np.sum(a2_delta, axis=0)
		# np.sum(a2_delta, axis=0).shape # (128,)
		self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
		self.b1 -= self.lr * np.sum(a1_delta, axis=0)

	def predict(self, data):
		self.x = data
		self.feedforward()
		return self.a3.argmax()

	def save(self, save_name):
		if not os.path.exists("./out/"):
			os.mkdir("./out/")
		if MODEL_OVER_WRIGHT or not os.path.exists("./out/{0}.pkl".format(save_name)):
			print("Saving Model ...")
			with open("./out/{0}.pkl".format(save_name), "wb") as file:
				pickle.dump(self, file)
		else:
			print("Saved Model Already Exists ...")
		return 0


def get_accuracy(model, data, lable, train=True):
	# model, data, lable = dnn_model, train_data, train_labels
	# model, data, lable = dnn_model, test_data, test_labels
	acc = 0
	for idx in tqdm(range(data.shape[0])):
		# idx = 0
		s = model.predict(data[idx])
		if s == np.argmax(lable[idx]):
			acc +=1
	acc = (acc/data.shape[0])*100
	if train:
		model.train_accuracy = acc
	else:
		model.test_accuracy = acc
	return acc



def tarin_test_model(is_train, is_test):

	dnn_model = None
	train_data, train_labels = None, None
	test_data, test_labels = None, None
	try:
		if is_train:
			train_data, train_labels = load_data(normalize=True)
			train_labels = get_one_hot_vector_array(train_labels)
			# train_data.shape
			dnn_model = MyNN(train_data, np.array(train_labels), 1500)
			for epoch in tqdm(range(dnn_model.epoch)):
				dnn_model.feedforward()
				dnn_model.backprop()
			print("Epoch: {0}, Loss: {1}".format(dnn_model.epoch, dnn_model.loss[-1]))
			# dnn_model.weight2
			print("Training accuracy : ", get_accuracy(dnn_model, train_data, np.array(train_labels)))
			dnn_model.save("mnist_dnn")
		if is_test:
			test_data, test_labels = load_data(train_data=False, normalize=True)
			# test_data.shape
			dnn_model = load_model("mnist_dnn")
			test_labels = get_one_hot_vector_array(test_labels)
			# dnn_model.weights_matrices
			if dnn_model != None:
				print("Training accuracy : ", get_accuracy(dnn_model, test_data, np.array(test_labels), False))
				dnn_model.save("mnist_dnn")
	except Exception as ex:
		print("Error:: {0}".format(ex))

	return 0



def main():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("train", type=int,help="1 to Train model, 0 to !Train model")
	parser.add_argument("test", type=int,help="1 to Test model, 0 to !Test model")
	args = parser.parse_args()

	if args.train and args.test:
		tarin_test_model(True, True)
	elif args.train:
		tarin_test_model(True, False)
	elif args.test:
		tarin_test_model(False, True)
	else:
		print("Thanks for wasiting your time!")
	return 0


if __name__ == '__main__':
	main()
