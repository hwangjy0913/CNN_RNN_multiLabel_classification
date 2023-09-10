#데이터셋 다운로드 및 압축 해제
  #현재 google com 내 워킹 디렉토리 위치 파악(이건 무조건 해야 됨!)
import os

current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

import requests
import tarfile

# 데이터셋 다운로드 URL
data_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"

# 다운로드할 경로 설정
download_dir = "/content"

# 다운로드 받을 디렉토리 생성
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# 데이터셋 다운로드
download_path = os.path.join(download_dir, "VOCtrainval_06-Nov-2007.tar")
response = requests.get(data_url, stream=True)
with open(download_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file.write(chunk)

# 압축 해제
with tarfile.open(download_path, "r") as tar:
    tar.extractall(download_dir)

# 압축 파일 삭제
os.remove(download_path)

print("PASCAL VOC 2007 데이터셋 다운로드 및 압축 해제 완료.")

#######
#전처리 시작
image_label=image_label_extract(50)#전체 샘플 뽑고 싶으면, 0 입력
print("multi_labels 개수 : ", len(image_label.multi_labels))
print(image_label.multi_labels)
print("images 개수 : ", len(image_label.images))

  #image 전처리
images=image_label.image_preprocessing(image_label.images)
multi_labels=image_label.multi_labels
print("multi_labels = ",multi_labels)
print("images.shape : ",images.shape)
#print("images : \n",images)

########
#모델 학습 및 사용
cnn_rnn = CNN_RNN(multi_labels,images,embedding_dimension = 10)
print("cnn_rnn.images.shape = ",cnn_rnn.images.shape)
print("cnn_rnn.embedded_sentences.shape = ",cnn_rnn.embedded_sentences.shape)
#정답 라벨(self.encoded_multi_labels_paded)은 shape = (sample 수(batch_size),문장 길이)로, 이 때, shape[1]차원에 들어가는 애들은 단어들이 정수 인코딩 된 상태!(sparse_categorical_cross entropy일 때)

cnn_rnn.compile(optimizer='rmsprop',loss="SparseCategoricalCrossentropy",metrics=["SparseCategoricalAccuracy"])
cnn_rnn.fit([cnn_rnn.embedded_sentences, cnn_rnn.images], cnn_rnn.encoded_multi_labels_paded,epochs=3,batch_size=64)


prediction = cnn_rnn([cnn_rnn.embedded_sentences, cnn_rnn.images])
print("prediction = \n",prediction)
print("prediction.shape = ",prediction.shape)#rnn 관련 모델 훈련시, 예측치 shape = (sample수(or batch size),문장 길이, vocab_size=num_classes(분류할 클래스 수))여야 한다.
print("embedding_table shape = ",cnn_rnn.pretrained_embedding_matrix.shape)
print("어휘사전 : ",cnn_rnn.idx2vocab)
print("정답 label : ",cnn_rnn.encoded_multi_labels_paded)
print("multi_labels = ",multi_labels)
#print(prediction)
