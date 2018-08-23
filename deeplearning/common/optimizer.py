import numpy as np

# https://arxiv.org/pdf/1412.6980.pdf  Adam paper
class Adam:
	def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.t = 0
		self.m = None
		self.v = None
		self.epsilon = 1e-08
	
	def update(self, trainable_layers):
		self.t += 1
		
		if self.m is None:
			weight = {}
			bias = {}
			for index, layer in enumerate(trainable_layers):
				weight[index] = np.zeros_like(layer.dw)
				bias[index] = np.zeros_like(layer.db)
			self.m = [weight, bias]
			self.v = [weight.copy(), bias.copy()]
	
		beta1_t = self.beta1 ** self.t # power t
		beta2_t = self.beta2 ** self.t # power t

		for index, layer in enumerate(trainable_layers):
			self.m[0][index] = self.beta1*self.m[0][index] + (1-self.beta1)*layer.dw #shape 안맞는 곱셈은 numpy broadcast로 해결됨
			self.m[1][index] = self.beta1*self.m[1][index] + (1-self.beta1)*layer.db 
			self.v[0][index] = self.beta2*self.v[0][index] + (1-self.beta2)*np.square(layer.dw)
			self.v[1][index] = self.beta2*self.v[1][index] + (1-self.beta2)*np.square(layer.db) 

			m_hat_weight = self.m[0][index] / (1-beta1_t)
			m_hat_bias = self.m[1][index] / (1-beta1_t)
			v_hat_weight = self.v[0][index] / (1-beta2_t)
			v_hat_bias = self.v[1][index] / (1-beta2_t)

			layer.w -= self.lr * m_hat_weight / (np.sqrt(v_hat_weight) + self.epsilon)
			layer.b -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)


class GradientDescent:
	def __init__(self, lr=0.001):
		self.lr = lr

	def update(self, trainable_layers):
		for layer in trainable_layers:
			layer.w -= self.lr * layer.dw
			layer.b -= self.lr * layer.db


