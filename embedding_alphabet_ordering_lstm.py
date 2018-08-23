import numpy as np
import deeplearning.layer as nn
import deeplearning.common.initializer as init
import deeplearning.common.model as net
import deeplearning.common.util as util
import deeplearning.common.optimizer as optimizer
import csv

x_len = 4

def read_csv(path):
	data = []
	with open(path, 'r', newline='') as f:
		wr = csv.reader(f)
		for line in wr:
			data.append(line)
	return np.array(data, np.int32)

def train(model, data):
	batch_size = 128
	loss = 0
	np.random.shuffle(data)

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :x_len]
		target_ = batch[:, -x_len:]
		onehot_target_ = np.eye(x_len)[target_.flatten()]

		logits = model.forward(input_, is_train=True)
		train_loss = model.backward(logits, onehot_target_)
		loss += train_loss
		
	return loss/len(data)


def validation(model, data):
	batch_size = 128
	loss = 0
	
	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :x_len]
		target_ = batch[:, -x_len:]
		onehot_target_ = np.eye(x_len)[target_.flatten()]
	
		logits = model.forward(input_, is_train=False)
		vali_loss = model.get_loss(logits, onehot_target_)
		loss += vali_loss
	return loss/len(data)


def test(model, data):
	batch_size = 128
	correct = 0

	for i in range( int(np.ceil(len(data)/batch_size)) ):
		batch = data[batch_size * i: batch_size * (i + 1)]
		input_ = batch[:, :x_len]
		target_ = batch[:, -x_len:]
		onehot_target_ = np.eye(x_len)[target_.flatten()] # [N*x_len, x_len]

		logits = model.forward(input_, is_train=False) # [N*output_num(=1)*x_len, x_len]
		
		equal = np.equal(logits.argmax(axis=1).reshape(-1, x_len), target_).astype(np.int32) # [N, x_len]
		equal = np.mean(equal, axis=1) # [N]
		equal[equal<1] = 0 #[N] 1이 아닌것 제외. 다 맞아야 1임.
		correct += np.sum(equal)

	return correct / len(data)


def run(model, train_set, vali_set, test_set):
	for epoch in range(1, 100+1):
		train_loss = train(model, train_set)
		vali_loss = validation(model, vali_set)
		accuracy = test(model, test_set)

		print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy)


lr = 0.001
hidden_size = 256 
output_num = 1
word_size = 26
embedding_size = 32
print('alphabet_ordering_LSTM using Embedding, learning_rate=', lr)

train_set = read_csv('./gen_data_for_embedding_lstm/train.csv')
vali_set = read_csv('./gen_data_for_embedding_lstm/vali.csv')
test_set = read_csv('./gen_data_for_embedding_lstm/test.csv')

model = net.model(optimizer.Adam(lr=lr))
#model = net.model(optimizer.GradientDescent(lr=lr))  

model.add(nn.embedding(word_size=word_size, embedding_size=embedding_size)) # [N, x_len, embedding_size]
model.add(nn.lstm(hidden_size=hidden_size, output_num=output_num)) # [N, output_num, hidden_size]
model.add(nn.reshape([-1, hidden_size])) # [N*output_num, hidden_size]
model.add(nn.affine(out_dim=16, w_init=init.xavier)) #[N*output_num, x_len*x_len]
model.add(nn.reshape([-1, 4])) # [N*output_num*x_len, x_len]

model.add_loss(nn.softmax_cross_entropy_with_logits())

run(model, train_set, vali_set, test_set)

#test
testdata = test_set[:10, :x_len]
pred = model.forward(testdata, is_train=False) # [N*output_num(=1)*x_len, x_len]
pred = pred.argmax(axis=1).reshape(-1, x_len) # [N*1*x_len, 1] => [N, x_len]

print("testdata_sample ==> pred")
print(np.hstack((testdata, np.full([10,1], '==>'), pred))) 
