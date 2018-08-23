# Numpy_RecurrentNeuralNetwork
Numpy_RecurrentNeuralNetwork

Numpy로 구현한 Recurrent Neural Network  
RNN 대신 LSTM만 구현함.  
Computational graph를 이용한 Backpropagation  

## deeplearning/
    * layer.py
        * Numpy로 구현한 함수들의 모음
            * FullyConnected Neural Network, Convolutional Neural Network, Recurent Neural Network(LSTM)
            * Embedding, Activation functions, Dropout, softmax_cross_entropy_with_logits 등등 구현
    * common/initializer.py, model.py, optimizer,py, util.py  
        * 학습에 사용되는 코드들의 모음
            * AdamOptimizer, GradientDescentOptimizer 등등 구현
        
## gen_data_for_embedding_lstm/
    * gen_data.py
        * embedding_alphabet_ordering_lstm.py 에서 사용되는 데이터 생성 코드
        * alphabet 4개와, 4개의 순서를 정답으로 생성
            * alphabet: [a, c, z, b] ==> ordering: [0, 4, 1, 2]  == [a, b, c, z] 라는 의미
            * alphabet은 숫자로 치환해서 사용 (a:0, b:1, ..., z:25)
          
## embedding_alphabet_ordering_lstm.py
    * alphabet의 ordering을 판별
    * Many to one LSTM + Fully Connected Neural Network -> logits을 4등분하여 학습 및 추론
    * gen_data_for_embedding_lstm/gen_data.py 에서 생성된 데이터 사용
        * 입력 데이터는 embedding처리, 정답 데이터는 one-hot-encodding
    
## many_to_many_hello_lstm.py
    * hell 을 넣으면 ello를 추론하는 모델
    * Many to many LSTM + Fully Connected Neural Network
    
## multi_layer_mnist_lstm.py
    * MNIST dataset 학습 및 추론하는 모델
    * Multi_layer(stacked) LSTM + Fully Connected Neural Network

## single_layer_mnist_lstm.py
    * MNIST dataset 학습 및 추론하는 모델
    * Single_layer LSTM + Fully Connected Neural Network

