import numpy as np
import deeplearning.layer as nn
import deeplearning.common.initializer as init
import deeplearning.common.model as net
import deeplearning.common.util as util
import deeplearning.common.optimizer as optimizer

lr = 0.1
hidden_size = 8 
output_num = 4
print('many_to_many_hello_LSTM, learning_rate=', lr)

model = net.model(optimizer.Adam(lr=lr))
#model = net.model(optimizer.GradientDescent(lr=lr))  

model.add(nn.lstm(hidden_size=hidden_size, output_num=output_num)) # [N, output_num, hidden_size]
model.add(nn.reshape([-1, hidden_size])) # [N*output_num, hidden_size]
model.add(nn.affine(out_dim=4, w_init=init.xavier)) #[N*output_num, 4]

model.add_loss(nn.softmax_cross_entropy_with_logits())

x = np.array([[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0]]]) # 'h', 'e', 'l', 'o' #[1, 4, 4]
y = np.array([[0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]) # 'e', 'l', 'l', 'o' #[4, 4]

for epoch in range(1, 50):
	logits = model.forward(x, is_train=True) # [N*output_num, 4]
	train_loss = model.backward(logits, y)

	pred = logits.argmax(axis=1)

	print("epoch:", epoch, "\ttrain_loss:", train_loss, "\tpred:", pred)
	