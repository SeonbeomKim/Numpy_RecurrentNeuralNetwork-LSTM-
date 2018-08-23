import numpy as np

class model:
	def __init__(self, optimizer):
		self.optimizer = optimizer
		self.layers = []
		self.loss_layer = None
		self.trainable_layers = []
	
	# layer 저장 및 학습 가능 layer 저장.
	def add(self, layer):
		self.layers.append(layer)
		
		#if trainable layer
		if 'w' in layer.__dict__:
			self.trainable_layers.append(layer)
		print(layer.__class__.__name__)

	# loss layer 저장.
	def add_loss(self, layer):
		self.loss_layer = layer


	# forward 연산. dropout인 경우에는 is_train 파라미터 추가됨.
	def forward(self, x, is_train=True):
		logits = np.array(x)
		for layer in self.layers:
			if layer.__class__.__name__ == 'dropout':
				logits = layer.forward(logits, is_train)
			else:
				logits = layer.forward(logits)
		return logits


	# 예측값과 타겟값의 오차 계산.
	def get_loss(self, logits, y):
		loss = self.loss_layer.forward(logits, y)
		return loss


	# backpropagation
	def backward(self, logits, y):
		loss = self.get_loss(logits, np.array(y)) #self.pred, self.target을 할당하기 위한 목적

		grad = self.loss_layer.backward() # loss layer gradient
		for index in reversed(range(len(self.layers))):
			grad = self.layers[index].backward(grad) #grad 계산하면 dw,db 갱신됨.

		self.optimizer.update(self.trainable_layers)
		return loss


	#예측값과 타겟값이 동일한지 체크
	def correct(self, logits, y, axis=1):
		compare = (np.argmax(logits, axis) == np.argmax(y, axis))
		return np.sum(compare)