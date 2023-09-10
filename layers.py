from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams#skipgrams 함수는 Skip-gram 모델을 학습하기 위해 필요한 학습 데이터를 생성하는 역할 -> 즉, SGNS 데이터셋을 만드는 과정 참고해보면,
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Dense, LSTM, Layer, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences#문장 패딩해주는 애
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

class word_processing_layer():
  def __init__(self,sentences,images,embedding_dimension):#위에서 sentences는 multi_labels 의미. 즉, 토큰화까지 완료되어 각 문장(문장 내 단어 모음)들이 리스트로 들어있는 리스트 / images는 패딩까지 와료된 numpy
    self.multi_labels = sentences
    self.images=images
      #2-1-1 : sentences의 원소들 중, None or []이 있는 index 뽑기 그 후, 해당 index 제거! 이에 대응되는 image도 전부 지워줘야 됨!
    indices_of_none_or_x = [index for index, item in enumerate(self.multi_labels) if item is None or item==[]]
    self.multi_labels = [multi_labels[index] for index in range(len(self.multi_labels)) if index not in indices_of_none_or_x]
    self.images = [self.images[index] for index in range(len(self.images)) if index not in indices_of_none_or_x]#NumPy에서 len() 함수는 다차원 배열의 경우 첫 번째 차원(가장 바깥 차원)의 길이를 반환

      #2-1-2 : multi_labels에서 object 1개짜리만 있는 img들의 index 확인 및 multi_labels, images의 image 제거(중심-주변 단어 embedding(SGNS) 할 때, 이들은 의미가 없기에)
    indices_of_1 = [index for index, item in enumerate(self.multi_labels) if len(item)==1]
    self.multi_labels = [multi_labels[index] for index in range(len(self.multi_labels)) if index not in indices_of_1]
    self.images = [self.images[index] for index in range(len(self.images)) if index not in indices_of_1]
    self.images = np.array(self.images)


      #2-2 : 정수 인코딩
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(self.multi_labels)#원래는 여기에 토큰화까지 완료된 sentence 집합(각 sentence들은 list들이 되어 있고, 이 list들이 하나의 list로 묶여있는 데이터)을 입력하는 거지만, 여기선 multi_label이 이에 해당
    self.vocab = tokenizer.word_index#단어 집합(어휘사전)

      #문장(multi_labels)들을 정수 인코딩된 형태로 나타내어 저장하기
    self.encoded_multi_labels=tokenizer.texts_to_sequences(self.multi_labels)

    self.vocab_size = len(self.vocab) + 1  # 단어 집합(vocabulary)의 크기를 어휘 사전(word2idx)의 길이에 1을 더한 이유는 일반적으로 자연어 처리 모델에서 사용되는 토큰 중 하나를 예약(reserved)하기 위해서입니다. 이 토큰은 텍스트에서 특정 단어가 아닌 "알 수 없는 단어" 또는 "패딩"을 나타내는데 사용됩니다.
    print('단어 집합의 크기 :', self.vocab_size)
    self.idx2vocab = {value : key for key, value in self.vocab.items()}#어휘사전 vocab은 key가 단어 즉, string이라서, 이후, 네거티브 샘플링이 잘 되었는지(데이터 셋이 올바르게 중심, 주변 단어로 짝지어져 있는지) 단어로 확인하려면, indexing으로 확인해야 하지만, key가 string, value가 index인 상태라 불가능하다. -> key-value 관계를 반대로 할 필요!
    self.idx2vocab[0] = "0"#나중에 예측 문장 확인할 때 0을 바꿀 때 사용 (실제로 SGNS의 embedding matrix에서 0행은 제로패딩시의 token의 embedding vector이기에!!!)
    print("idx2vocab = ",self.idx2vocab)


    #2-3 SGNS
      #2-3-1 네거티브 샘플링
    self.skip_grams = [skipgrams(sample, vocabulary_size=self.vocab_size, window_size=2) for sample in self.encoded_multi_labels]#skipgrams 함수는 Skip-gram 모델을 학습하기 위해 필요한 학습 데이터를 생성하는 역할 -> 즉, SGNS 데이터셋을 만드는 과정 참고해보면,
          #네거티브 샘플링 제대로 되었는지 SGNS 데이터셋 확인(이 파일에선 코드 생략 ->PASCAL VOC 2007 데이터 전처리(img - labels).ipynb 참고)

      #2-3-1 : SGNS 모델 구성
    input_center = Input(shape=(1,), dtype=tf.int32)
    input_context = Input(shape=(1,), dtype=tf.int32)
    embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=embedding_dimension)
    center_embedding = embedding_layer(input_center)#중심단어(input_center)의 정수 인코딩(or one-hot)에 해당하는 embedding table의 행 객체 의미?(print해보면, 느낌 옴.)
    context_embedding = embedding_layer(input_context)
    dot_product = Dot(axes=2)([center_embedding, context_embedding])
    output = Dense(1, activation='sigmoid')(dot_product)
    model = tf.keras.Model(inputs=[input_center, input_context], outputs=output)

        # 손실 함수 정의 및 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #SGNS 모델 학습
    for epoch in range(1, 6):
        loss = 0
        for _, elem in enumerate(self.skip_grams):#skip_gram 구조를 보면, 알 수 있듯, elem은 하나의 토큰화되어 정수인코딩된 문장의 각 중심,주변 단어 쌍들을 list로 모아 하나의 list로/ 그리고, 각 쌍들에 대한 정답 label을 모다 list로 만들어 이 두 list를 쌍으로 갖는 튜플이다.
            first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
            second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
            labels = np.array(elem[1], dtype='int32')
            X = [first_elem.reshape(-1,1), second_elem.reshape(-1,1)]
            Y = labels.reshape(-1,1)
            loss += model.train_on_batch(X,Y)[0]
            accuracy=  model.train_on_batch(X,Y)[1]
        print('Epoch :',epoch, 'Loss :',loss, 'accuracy : ',accuracy)

    # 단어 임베딩 추출
    self.word_embeddings = embedding_layer.get_weights()[0]#이 행렬이 바로 embedding table로 각 행이 단어들의 embedding vector이고, 논문의 Ul 행렬에 해당!
    print("embedding tabel shape : ",self.word_embeddings.shape)
    print(type(self.word_embeddings))

    # 2-4 패딩
    max_len=max(len(review) for review in self.encoded_multi_labels)
    self.encoded_multi_labels_paded = pad_sequences(self.encoded_multi_labels, maxlen=max_len,padding='post')#정수 인코딩된 문장 리스트들을 최대 문장길이로 패딩하여 각 문장 길이를 맞춘 것 뿐!(list타입!) /padding='post' -> 0인 숫자들로 패딩을 할 때, 패딩은 뒷부분에서 이뤄지도록 함.

    #문장 embedding
  def __call__(self):#출력값으로, 어휘사전(self.vocab,self.idx2vocab),embedding table, embedding된 문장 집합(numpy 샘플 문장 수*문장 길이*embedding 차원 형태 = lstm에 들어갈 input 형태)=embedding layer 통과시킨 값
    # 사전 훈련된 임베딩 행렬(self.word_embeddings)을 사용하여 Embedding 레이어 초기화
    pretrained_embedding_matrix = self.word_embeddings
    vocab_size = pretrained_embedding_matrix.shape[0]#행의 개수 = padding 토큰 포함 vocab 총 개수 (어휘사전 index = 행 index)
    embedding_dim = pretrained_embedding_matrix.shape[1]#각 단어들의 embedding vector 크기
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[pretrained_embedding_matrix],
        trainable=False)  # 임베딩 행렬을 고정

    # 입력 문장을 임베딩 벡터로 변환
    embedded_sentences = embedding_layer(self.encoded_multi_labels_paded)# embedded_sentences은 입력 문장에 대한 임베딩 벡터를 담고 있는 텐서입니다.
    print("self.encoded_multi_labels_paded.shape = ", self.encoded_multi_labels_paded.shape)
    print("embedded_sentences.shape",embedded_sentences.shape)
    return self.images, self.vocab, self.encoded_multi_labels, self.encoded_multi_labels_paded ,self.idx2vocab,pretrained_embedding_matrix,embedded_sentences#embedded_sentences가 lstm에 들어갈 input / self.encoded_multi_labels_padded는 모델의 정답

class projection_layer(Layer):
  def __init__(self,img_feature_size,pretrained_embedding_matrix_shape):#vgg를 거쳐 output layer에서 softmax를 거친 각 img_feature는 1*1000 행벡터 and 그럼 덧셈 연산을 위해 hidden state size도 1000으로 위 lstm에서 맞췄다
    super().__init__()
#img_feature은 shape이 sample img 개수*feature vector 크기
    self.weights_img = tf.Variable(tf.random.normal([img_feature_size[1], pretrained_embedding_matrix_shape[1]]), trainable=True)#pretrained_embedding_matrix_shape[1]는 embedding table의 열 수=embedding size
    self.weights_hidden_states = tf.Variable(tf.random.normal([img_feature_size[1], pretrained_embedding_matrix_shape[1]]), trainable=True)#trainable=True로 설정하면 이 변수들은 모델 학습 중에 업데이트되며, 학습 중에 최적화되는 매개변수로 사용/img_feature_size=hidden state size로 이미 위에서 맞춤!
    self.bias_proj = tf.Variable(tf.zeros([pretrained_embedding_matrix_shape[1]]), trainable=True)
    self.relu = Activation("relu")

  def __call__(self,image_features,sentence_hidden_states):#논문과 모델 구조를 보면 알겠지만, projection layer에는 하나의 image_feature 당 for문으로 해당 하나의 sentence의 단어에 대한 hidden state 순서대로 넣어야한다! -> 각 label probability구할 때까지 이 과정 유지!

    #tf.matmul은 고차원 tensor 내적에서도 그대로 사용가능하며, (a,b,c,d)인 tensor A와 (d,e) B행렬의 곱을 예로 들면, A에는 (c,d)행렬이 모여있는 것으로 보고, 각(c,d)와 (d,e)가 내적 연산이 이뤄져 결과는 (a,b,c,e)이다.
    X=tf.matmul(image_features,self.weights_img)[:,tf.newaxis,:]+tf.matmul(sentence_hidden_states,self.weights_hidden_states)#tf.matmul(image_features,self.weights_img) shape = (sample 수, emb_size), tf.matmul(sentence_hidden_states,self.weights_hidden_states) shape = (sample 수, 문장길이, emb_size)이기에,
    #이 둘을 바로 덧셈은 불가능하여 tf.matmul(image_features,self.weights_img) shape = (sample 수, emb_size)을 (sample 수, 1, emb_size)로 변환하여(broadcasting) 연산해야 됨
    X=self.relu(X)
    print("X shape = ",X.shape)
    return X

class prediction_layer(Layer):
  def __init__(self,embedding_matrix):#embedding_matrix(위 셀의 pretrained_embedding_matrix)의 transpose가 여기서의 weight matrix
    super().__init__()

    self.fixed_weights=tf.constant(np.transpose(embedding_matrix))
    self.bias = tf.Variable(tf.zeros([1,self.fixed_weights.shape[1]]), trainable=True)
    self.softmax=Activation("softmax")
    self.predicted_label_int=list([])

  def __call__(self,X):#X shape = (sample 수, 문장 내 단어 수, embedding size)로, projection layer의 output
    Y=tf.matmul(X,self.fixed_weights)+self.bias
    Y=self.softmax(Y)
    print("Y.shape = ",Y.shape)#Y shape = (sample, 각 문장 단어 수(단어 길이), vocab_size)
    return Y#모델의 정답은 밑의 주석처럼 이 확률의 각 행에서 예측한 단어를 찾아내야 한다!(decoding_layer)