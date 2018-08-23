import numpy as np
import deeplearning.layer as nn
import deeplearning.common.initializer as init
import deeplearning.common.model as net
import deeplearning.common.util as util
import deeplearning.common.optimizer as optimizer
from tensorflow.examples.tutorials.mnist import input_data #for MNIST dataset

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = np.hstack((mnist.train.images, mnist.train.labels)) #shape = 55000, 794   => 784개는 입력, 10개는 정답.
vali_set = np.hstack((mnist.validation.images, mnist.validation.labels))
test_set = np.hstack((mnist.test.images, mnist.test.labels))


def train(model, data):
	batch_size = 128
	loss = 0
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784].reshape(-1, 28, 28) # [N, T, x_len] N: batch, T: 28, x_len: 28
		target_ = batch[:, 784:]

		logits = model.forward(input_, is_train=True)
		train_loss = model.backward(logits, target_)
		loss += train_loss
		
	return loss/len(data)


def validation(model, data):
	batch_size = 128
	loss = 0
	
	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784].reshape(-1, 28, 28)
		target_ = batch[:, 784:]
	
		logits = model.forward(input_, is_train=False)
		vali_loss = model.get_loss(logits, target_)
		loss += vali_loss
	return loss/len(data)


def test(model, data):
	batch_size = 128
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :784].reshape(-1, 28, 28)
		target_ = batch[:, 784:]

		logits = model.forward(input_, is_train=False)
		check = model.correct(logits, target_, axis=1)
		correct += check
	return correct/len(data)


def run(model, train_set, vali_set, test_set):
	for epoch in range(1, 300):
		train_loss = train(model, train_set)
		vali_loss = validation(model, vali_set)
		accuracy = test(model, test_set)

		print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy)


lr = 0.0001
hidden_size = 256
output_num = 1
print('multi_layer_mnist_LSTM, learning_rate=', lr)

model = net.model(optimizer.Adam(lr=lr))
#model = net.model(optimizer.GradientDescent(lr=lr)) 
	
model.add(nn.lstm(hidden_size=hidden_size)) # [N, T, hidden_size]
model.add(nn.lstm(hidden_size=hidden_size)) # [N, T, hidden_size]
model.add(nn.lstm(hidden_size=hidden_size, output_num=output_num)) # [N, output_num, hidden_size]
model.add(nn.reshape([-1, hidden_size])) # [N*output_num, hidden_size]
model.add(nn.affine(out_dim=10, w_init=init.xavier)) #[N*output_num, 10]

model.add_loss(nn.softmax_cross_entropy_with_logits())


run(model, train_set, vali_set, test_set)

