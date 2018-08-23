import deeplearning.common.util as util
import deeplearning.common.initializer as init
import numpy as np

class relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		mask = (x<=0) # 0이하인 값 mask,  if x = [10, -1, 3] => mask = [false, true, false]
		self.mask = mask # backward할 때, mask된 부분(0이하인 값)은 미분 0
		x[mask] = 0 # 0이하인 값에 0 할당.
		return x

	def backward(self, grad):
		grad[self.mask] = 0 #forward 값이 0이하였던 부분은 미분값 0으로 할당.
		return grad

class sigmoid:
	def __init__(self):
		self.sigvalue = None 

	def forward(self, x):
		sigvalue = 1/(1+np.exp(-x))
		self.sigvalue = sigvalue
		return sigvalue

	def backward(self, grad=1):
		return grad*self.sigvalue*(1-self.sigvalue)

class tanh:
	def __init__(self):
		self.tanhvalue = None 

	def forward(self, x):
		tanhvalue = (np.exp(2*x)-1) / (np.exp(2*x)+1) 
		self.tanhvalue = tanhvalue
		return tanhvalue

	def backward(self, grad=1):
		return grad* (1-self.tanhvalue) * (1+self.tanhvalue)

class flatten:
	def __init__(self):
		self.x_shape = None

	def forward(self, x):
		self.x_shape = x.shape # [N, ~]
		x = x.reshape(self.x_shape[0], -1)
		return x

	def backward(self, grad):
		grad = grad.reshape(self.x_shape)
		return grad

class reshape:
	def __init__(self, shape):
		self.want_shape = shape
		self.x_shape = None #original shape

	def forward(self, x):
		self.x_shape = x.shape 
		x = x.reshape(self.want_shape)
		return x

	def backward(self, grad):
		grad = grad.reshape(self.x_shape)
		return grad

class transpose:
	def __init__(self, shape):
		self.want_shape = shape
		self.recover_shape = None #original shape

	def forward(self, x): 
		self.recover_shape = np.argsort(self.want_shape) #want_shape: [0, 3, 1, 2] => recover_shape: [0, 2, 3, 1]
		# [a, b, c, d].transpose([0, 3, 1, 2]) => [a, d, b, c],   [a, d, b, c].transpose([0, 2, 3, 1]) => [a, b, c, d]
		x = x.transpose(self.want_shape)
		return x

	def backward(self, grad):
		grad = grad.transpose(self.recover_shape)
		return grad

class embedding:
	def __init__(self, word_size, embedding_size):
		self.word_size = word_size
		self.embedding_size = embedding_size
		self.x = None

		self.w = np.random.randn(word_size, embedding_size)
		self.b = 0 #안쓰임. optimizer 에서 b도 업데이트 하기 때문에 에러 안나도록 작성.

		self.dw = None
		self.db = 0 #안쓰임.

	def forward(self, x):
		x = np.array(x)
		self.x = x # [N, x_len]
		N, x_len = x.shape
		return self.w[x] # [N, x_len, embedding_size]

	def backward(self, grad):
		#grad: [N, x_len, embedding_size]
		dw = np.zeros([self.word_size, self.embedding_size]).astype(np.float32) # [word_size, embedding_size]
		x = self.x.reshape(-1) # [N*x_len]
		grad = grad.reshape(-1, self.embedding_size) # [N*x_len, embedding_size]

		for i, index in enumerate(x):
			dw[index] += grad[i] # i: 0 ~ N*x_len
		self.dw = dw

class lstm:
	def __init__(self, hidden_size, output_num=0, w_init=init.xavier, b_init=0):
		self.hidden_size = hidden_size
		self.output_num = output_num
		self.w_init = w_init

		self.w = None
		self.b = np.full(hidden_size*4, b_init).astype(np.float32)

		self.dw = None
		self.db = None

		self.x = None
		self.x_shape = None
		self.activation_values = None
		self.H = None
		self.H_shape = None

	def forward(self, x):
		self.x = x
		N, T, x_len = x.shape # [N, T, x_len]
		self.x_shape = np.array([N, T, x_len])

		H = np.zeros([N, self.hidden_size]) #initial H
		C = np.zeros([N, self.hidden_size]) #initial C

		hidden = []
		activation_values = []
		if self.w is None:
			w_x = self.w_init([x_len, self.hidden_size*4], x_len) #[x_len, hidden_size*4]
			w_h = self.w_init([self.hidden_size, self.hidden_size*4], self.hidden_size) #[hidden_size, hidden_size*4]
			self.w = np.vstack((w_x, w_h)) # [x_len+hidden_size, hidden_size*4]

		for time in range(T):
			x_t = x[:, time, :] # [N, x_len]
			values = np.dot(x_t, self.w[:x_len, :]) + np.dot(H, self.w[x_len:]) + self.b # [N, hidden_size*4]
			f = sigmoid().forward(values[:, :self.hidden_size])
			i = sigmoid().forward(values[:, self.hidden_size:self.hidden_size*2])
			o = sigmoid().forward(values[:, self.hidden_size*2:self.hidden_size*3])
			g = tanh().forward(values[:, self.hidden_size*3:self.hidden_size*4])

			C_t_minus_one = C
			C = C*f + i*g
			tanh_C = tanh().forward(C)
			H = tanh_C * o
			hidden.append(H)
			
			activation_values.append([f, i, o, g, C_t_minus_one, tanh_C])

		self.activation_values = np.array(activation_values) # [time, N, hidden_size]
		hidden = np.array(hidden) # [time, N, hidden_size]
		hidden = hidden.transpose(1, 0, 2) # [N, T, hidden_size]
		self.H = hidden # [N, T, hidden_size]
		self.H_shape = hidden.shape # [N, T, hidden_size]

		if self.output_num == 0 or self.output_num == T:
			return hidden   # [N, T, hidden_size]
		else:
			return hidden[:, -self.output_num:, : ] # [N, output_num, hidden_size]

	def backward(self, grad):
		#grad shape: [N, output_num, hidden_size]
		N, T, x_len = self.x_shape

		grad_mask = np.zeros(self.H.shape).astype(np.float32) # [N, T, hidden_size]
		grad_mask[:, -self.output_num:, :] = grad # [N, T, hidden_size] #output으로 사용안된부분은 0임.
		grad_mask = grad_mask.transpose(1, 0, 2) # [T, N, hidden_size]
		dx_mask = np.zeros(self.x_shape).astype(np.float32) # [N, T, x_len]
		
		grad_H = 0
		grad_C = 0
		dw = 0
		db = 0

		for time in reversed(range(T)):
			time_grad = grad_mask[time] # [N, hidden_size]
			grad_H = grad_H + time_grad # [N, hidden_size]

			f, i, o, g, C_t_minus_one, tanh_C = self.activation_values[time] # 전부 [N, hidden_size]
			df = f * (1-f) #sigmoid 미분 [N, hidden_size]
			di = i * (1-i) #sigmoid 미분 [N, hidden_size]
			do = o * (1-o) #sigmoid 미분 [N, hidden_size]
			dg = (1-g) * (1+g) #tanh 미분 [N, hidden_size]
			dtanh_C = (1-tanh_C) * (1+tanh_C) #tanh 미분 [N, hidden_size]

			for_f_i_g = (grad_H * o * dtanh_C) + grad_C # [N, hidden_size]
			temp_o = grad_H * tanh_C # [N, hidden_size] 

			#grad_C update
			grad_C = for_f_i_g * f # [N, hidden_size]

			temp_f = for_f_i_g * C_t_minus_one # [N, hidden_size]
			temp_i = for_f_i_g * g # [N, hidden_size]
			temp_g = for_f_i_g * i # [N, hidden_size]

			temp_f = temp_f * df # [N, hidden_size]
			temp_i = temp_i * di # [N, hidden_size]
			temp_o = temp_o * do # [N, hidden_size]
			temp_g = temp_g * dg # [N, hidden_size]

			values = np.hstack((temp_f,temp_i,temp_o,temp_g)) #가로방향으로 concat [N, hidden_size*4]
					
			#bias update
			db += np.sum(values, 0) # [hidden_size*4]

			w_x = self.w[:x_len, :] # [x_len, hidden_size*4]
			w_h = self.w[x_len:, :] # [hidden_size, hidden_size*4]
			x = self.x[:, time, :] # [N, x_len]
			H = self.H[:, time, :] # [N, hidden_size]

			dx = np.dot(values, w_x.T) # [N, x_len]
			dx_mask[:, time, :] = dx # [N, T, x_len]

			dw_x = np.dot(x.T, values) # [x_len, hidden_size*4]

			grad_H = np.dot(values, w_h.T) # [N, hidden_size]
			dw_h = np.dot(H.T, values) # [hidden_size, hidden_size*4]

			dw += np.vstack((dw_x, dw_h)) # [x_len+hidden_size, hidden_size*4]	

		self.dw = dw
		self.db = db
			
		return dx_mask


class conv2d:
	def __init__(self, filters, kernel_size, strides, w_init=init.xavier, b_init=0):
		self.filters = filters
		self.kernel_size = np.array(kernel_size)
		self.strides = np.array(strides)
		self.w_init = w_init

		self.x_shape = None # [N, H, W, C]
		self.out_shape = None #[H, W]
		self.pad = None
		self.col = None

		self.db = None
		self.dw = None

		self.w = None
		self.b = np.full(filters, b_init).astype(np.float32)

	def forward(self, x):
		N, H, W, C = x.shape
		self.x_shape = np.array([N, H, W, C])
		
		#weight init		
		if self.w is None:
			self.w = self.w_init([self.filters, C, *self.kernel_size], np.prod(self.kernel_size)*C) # [filters, in, FH, FW]
		
		out_shape = np.ceil( np.array([H, W]) / self.strides ).astype(np.int32)
		self.out_shape = out_shape

		pad = (out_shape * self.strides) - np.array([H, W]) + self.kernel_size - self.strides
		pad[pad<0] = 0 # max(0, pad)
		self.pad = pad		

		FH, FW = self.kernel_size

		col = util.im2col(x, self.kernel_size, self.strides, pad, out_shape) # [N*out_H*out_W, FH*FW*C]
		self.col = col
		w = self.w.reshape(self.filters, FH*FW*C).T # [FH*FW*C, filters] 

		out = np.dot(col, w) + self.b # [N*out_H*out_W, filters]
		out = out.reshape(N, *out_shape, self.filters) # [N, out_H, out_W, filters]
		return out

	def backward(self, grad):
		#if grad.ndim == 2: #2차원인 경우 FCNN 계층에서 넘어왔다는 것이고, [N, out_H*out_W*filters] 의 shape를 가질것.
		#	grad = grad.reshape(self.x_shape[0], *self.out_shape, self.filters) # [N, out_H, out_W, filters]

		N, out_H, out_W, filters = grad.shape
		FH, FW = self.kernel_size

		grad = grad.reshape(N*out_H*out_W, filters) # [N*out_H*out_W, filters]
		w = self.w.reshape(filters, -1).T # [FH*FW*C, filters]
		#self.w: [filters, C, FH, FW]
		#self.col: [N*out_H*out_W, FH*FW*C]
		#origin_x: [N, H, W, C]

		self.db = np.sum(grad, axis=0) # [filters]

		dw = np.dot(self.col.T, grad) # [FH*FW*C, filters] <= [FH*FW*C, N*out_H*out_W] * [N*out_H*out_W, filters]
		dw = dw.T # [filters, FH*FW*C] # self.w shape과 맞춰주기 위해서 transpose 하고 reshape.
		self.dw = dw.reshape(filters, -1, FH, FW) # [filters, C, FH, FW]

		dx = np.dot(grad, w.T) # [N*out_H*out_W, FH*FW*C] <= [N*out_H*out_W, filters] * [filters, FH*FW*C]
		dx = util.col2im(dx, self.x_shape, self.kernel_size, self.strides, self.pad, self.out_shape)
		return dx # [*self.x_shape] == forward때 입력되었던 x의 shape [N, H, W, C]
		
class maxpool2d:
	def __init__(self, kernel_size, strides):
		self.kernel_size = np.array(kernel_size)
		self.strides = np.array(strides)

		self.x_shape = None # [N, H, W, C]
		self.out_shape = None #[H, W]
		self.pad = None
		self.max_mask = None

	def forward(self, x):
		N, H, W, C = x.shape
		self.x_shape = np.array([N, H, W, C])

		out_shape = np.ceil( np.array([H, W]) / self.strides ).astype(np.int32)
		self.out_shape = out_shape

		pad = (out_shape * self.strides) - np.array([H, W]) + self.kernel_size - self.strides
		pad[pad<0] = 0 # max(0, pad)
		self.pad = pad		

		FH, FW = self.kernel_size

		col = util.im2col(x, self.kernel_size, self.strides, pad, out_shape) # [N*out_H*out_W, FH*FW*C]
		col = col.reshape(N*np.prod(out_shape)*C, FH*FW) # [N*out_H*out_W*C, FH*FW]
				
		max_mask = np.argmax(col, axis=1) # [N*out_H*out_W*C]
		self.max_mask = np.eye(FH*FW)[max_mask] # [N*out_H*out_W*C, FH*FW]
		
		max_col = np.max(col, axis=1) # [N*out_H*out_W*C]
		max_col = max_col.reshape(N, *out_shape, C) # [N, out_H, out_W, C]
		return max_col

	def backward(self, grad):
		#if grad.ndim == 2: #2차원인 경우 FCNN 계층에서 넘어왔다는 것이고, [N, out_H*out_W*filters] 의 shape를 가질것.
		#	grad = grad.reshape(self.x_shape[0], *self.out_shape, -1) # [N, out_H, out_W, filters]

		N, out_H, out_W, filters = grad.shape
		FH, FW = self.kernel_size		
		
		grad = grad.reshape(N*out_H*out_W*filters, 1) # [N*out_H*out_W*filters, 1]
		col = self.max_mask*grad # [N*out_H*out_W*filters, FH*FW] <= [N*out_H*out_W*filters, FH*FW] * broadcast([N*out_H*out_W*filters, 1])
		col = col.reshape(N*out_H*out_W, FH*FW*filters) # [N*out_H*out_W, FH*FW*filters]

		x = util.col2im(col, self.x_shape, self.kernel_size, self.strides, self.pad, self.out_shape)
		return x # [*self.x_shape] == forward때 입력되었던 x의 shape [N, H, W, C]

class dropout:
	def __init__(self, keep_prob):
		self.keep_prob = keep_prob
		self.mask = None

	def forward(self, x, is_train=True):
		if is_train == True:
			uniform = np.random.uniform(0, 1, size=x.shape) # [0, 1)
			mask = uniform > self.keep_prob # keep_prob보다 작으면 false, 크면 true
			#0.6으로 치면 0.6보다 작은 값들(==60%)는 false, 0.6보다 큰 값들(==40%)는 true
			self.mask = mask #즉 mask는 지울 값들을 true로 mask 함.
			x[mask] = 0 #true인 위치의 값을 0으로 dropout.
			return x
		else:
			self.mask = np.full(x.shape, False)
			return x

	def backward(self, grad):
		grad[self.mask] = 0 #dropout 시켰던 부분은 미분값이 0이므로 grad도 0으로 할당.
		return grad

class affine:
	def __init__(self, out_dim, w_init=init.xavier, b_init=0):
		self.out_dim = out_dim
		self.w_init = w_init
		self.b = np.full(out_dim, b_init).astype(np.float32) # bias

		self.x = None # input
		self.w = None

		self.dw = None # w gradient
		self.db = None # bias gradient

	def forward(self, x):
		self.x = x # input
		in_dim = x.shape[-1]

		#weight init
		if self.w is None:
			self.w = self.w_init([in_dim, self.out_dim], in_dim) # [filters, in, FH, FW]

		out = np.matmul(x, self.w) + self.b
		return out

	def backward(self, grad=1):
		#x.T = [w_shape[0], batch], grad = [batch, w_shape[1]]  즉 np.matmul(x.T, grad) 하면 batch전체에 관해 dw가 계산됨.
		self.dw = np.matmul(self.x.T, grad) #shape 때문에 이렇게 됨. 계산그래프 그려보면 이해됨.
		self.db = np.mean(grad, axis=0) #batch별 평균.
		dx = np.matmul(grad, self.w.T) # x에 관해서 계속 backpropagation 되기 때문에 x에관한 미분을 리턴해서 이전 layer에 전파.
		return dx

class softmax_cross_entropy_with_logits:
	def __init__(self):
		self.target = None
		self.pred = None #softmax 결과
		self.loss = None 

	def forward(self, x, target):
		target = np.array(target)
		self.target = target

		#softmax
		max_value = np.max(x, axis=1, keepdims=True)
		exp = np.exp(x - max_value) #max값을 빼도, 빼지 않은 것과 결과는 동일하며, 빼지 않으면 값 overflow 발생 가능. 
		pred = exp / np.sum(exp, axis=1, keepdims=True)
		self.pred = pred

		#cross_entropy
		epsilon = 1e-07
		loss = -target*np.log(pred + epsilon) # pred가 0이면 np.log = -inf
		loss = np.mean(np.sum(loss, axis=1), axis=0) #data별로 sum 하고, batch별로 mean
		self.loss = loss
		return loss

	def backward(self, grad=1):
		#return np.mean(self.pred-self.target, axis=0) #배치별로 gradient 평균냄.
		return (self.pred-self.target)/self.target.shape[0] #배치사이즈로 나눠줌. 여기서 안나누고 affine.backward에서 나눠도되긴함
		#근데 affine.backward에서 나누면 affine 레이어마다 나눠야해서 계산량이 더 많음.

