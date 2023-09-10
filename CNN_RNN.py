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

class CNN_RNN(Model):
  def __init__(self,sentences,images_numpy,embedding_dimension):#여기서 images_numpy는 행의 수 = 224, 열의 수 = 224로 전처리가 되어 있어야 한다!(for vgg)-2번째 셀
    super().__init__()

    self.word_processing_layer=word_processing_layer(sentences,images_numpy,embedding_dimension=embedding_dimension)#layer 생성할 때, 이미 모델 학습까지 함! / embedding_dimension=15(더 크거나 작게 해도 됨), sentences=multi_labels, images_numpy=images
    self.images, self.vocab, self.encoded_multi_labels, self.encoded_multi_labels_paded ,self.idx2vocab,self.pretrained_embedding_matrix,self.embedded_sentences = self.word_processing_layer()#pretrained_embedding_matrix : embedding tabel / embedded_sentences : lstm에 들어갈 input값으로 embedding vector들로 나타낸 문장 집합
    self.vgg_model = VGG16(weights='imagenet')
    features=self.vgg_model(np.expand_dims(self.images[0,:,:,:],axis=0))
    self.lstm=LSTM(features.shape[1],return_sequences=True)#윗줄의 images의 shape=(image sample 수,행 수,열 수, ch 수) / 여기서 hidden state size = feature의 열 수여야 projection에서 덧셈 가능!
    self.projection_layer = projection_layer(features.shape,self.pretrained_embedding_matrix.shape)
    self.prediction_layer = prediction_layer(self.pretrained_embedding_matrix)

  def __call__(self,inputs,training=False):#모델 훈련시키기 위해선 "training=False" 꼭 써야됨! -> why? / #실제 실용적인 면에서 사용할 때는 여기서 훈련시킨 layer들을 beam_search_decoder로 데려가 거기서 쓸 것!/embedded_sentences : (샘플 수(문장 수),문장 길이(각 문장의 단어 수),embedding 차원) 이 형태로 lstm에 입력해야 한다!(즉, 각 행은 각 단어의 embedding vector!)/images : (sample 수, 행의 수 = 224, 열의 수 = 224, ch=3)으로, word_processing까지 거친 상태여야함.
    #inputs = [embedded_sentences,images]
    #이미지가 vgg지나 feature로
    images = preprocess_input(inputs[1])#입력 이미지를 VGG16 모델 또는 다른 일부 모델에 맞게 사전 처리(평균 값 제거/채널 정규화)
    features=self.vgg_model(images)#feature = (sample 수, class 수)
    #embedded_sentences -> lstm
    hidden_states=self.lstm(inputs[0]) #hidden_states = (sample 수(문장 개수),각 문장의 단어 개수, hidden state size) - 여기서 hidden state size = feature의 열 수여야 projection에서 덧셈 가능!

    #projection_layer
    X=self.projection_layer(features,hidden_states)#features = (sample 수, class 수) / hidden_states = (sample 수(문장 개수),각 문장의 단어 개수, hidden state size) -> projection_layer에서 입력값 형태 이렇게 유지하기!
                                                    #projection의 output shape= X shape = (sample 수, 문장 내 단어 수, embedding size)
    Y=self.prediction_layer(X)
    #decoding layer에서 정수 인코딩 필요(beam search decoder에서만 - 실용화용 새로운 함수 만들 것!)
    return Y