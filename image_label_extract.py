import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import tensorflow as tf
from collections import Counter

class image_label_extract():
  def __init__(self,sample_num):# sample_num은 내가 데이터셋에서 쓸 sample 수 - 전체 sample을 이용할 거면, 0
  # 데이터셋 디렉토리 설정
    dataset_dir = "/content/VOCdevkit/VOC2007"
    jpeg_images_dir = os.path.join(dataset_dir, "JPEGImages")#JPEGimage들이 있는 디렉터리 주소
    annotations_dir = os.path.join(dataset_dir, "Annotations")#각 image에 대응되는 주석들이 저장됨.

  # JPEGImages 디렉토리 내의 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(jpeg_images_dir) if f.endswith('.jpg')]

  #모든 image 샘플들을 하나의 집합으로(이후 4차원 Tensor로 만들 것) - 이 images list에는 numpy 배열 type의 image를 저장하게 할 것
    self.images=list([])

  #각 image에 대응되는 label들의 쌍을 하나의 집합(multi_labels)으로
    self.multi_labels=list([])#numpy로 하면, append를 할 때, 그냥 벡터의 각 entry로 분해되어 추가되기에, image와 대응되는 label들의 집합을 분류할 수가 없다. -> 이 multi_labels는 RNN들어갈 input용

    # 처음 sample_num개 이미지 처리(이번 코드와 이후의 코드를 보면, 각 이미지에 있는 객체들의 이름이 모두 labeling 되어 있음을 알 수 있다.)
    if sample_num==0:#0이면 전체 샘플을 사용할 것이라는 의미
      for image_file in image_files:
        image_path = os.path.join(jpeg_images_dir, image_file)#각 image들의 주소(위치)
        labels=list([])#각 image에 들어있는 모든 object name들이 들어갈 것

      # 이미지 출력
        image = Image.open(image_path)#해당 주소의 image 객체 생성

      # 해당 이미지에 대응하는 주석 파일 불러오기
        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        annotation_path = os.path.join(annotations_dir, annotation_file)

      # 주석 파일 파싱
        tree = ET.parse(annotation_path)
        root = tree.getroot()

      # 객체 정보 출력(이미지 내 존재하는 객체들의 이름 출력)
        for obj in root.findall('object'):#각 image에 대응되는 주석 파일에 있는 object들을 하나씩 불러오는 것.(즉, image에 있는 object를 ....)
          name = obj.find('name').text
          labels.append(name)
        image_array = np.array(image)
        self.images.append(image_array)#numpy에서의 append는 image shape이 모두 동일해야 가능 -> images를 list로 하고, 이들을 tensorflow에서 padding을 통해 shape 맞추기
        self.multi_labels.append(labels)#각각의 labels들이 묶음이 되도록 만들기
    else:
      for image_file in image_files[:sample_num]:
        image_path = os.path.join(jpeg_images_dir, image_file)#각 image들의 주소(위치)
        labels=list([])#각 image에 들어있는 모든 object name들이 들어갈 것

      # 이미지 출력
        image = Image.open(image_path)#해당 주소의 image 객체 생성

      # 해당 이미지에 대응하는 주석 파일 불러오기
        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        annotation_path = os.path.join(annotations_dir, annotation_file)

      # 주석 파일 파싱
        tree = ET.parse(annotation_path)
        root = tree.getroot()

      # 객체 정보 출력(이미지 내 존재하는 객체들의 이름 출력)
        for obj in root.findall('object'):#각 image에 대응되는 주석 파일에 있는 object들을 하나씩 불러오는 것.(즉, image에 있는 object를 ....)
          name = obj.find('name').text
          labels.append(name)
        image_array = np.array(image)
        self.images.append(image_array)#numpy에서의 append는 image shape이 모두 동일해야 가능 -> images를 list로 하고, 이들을 tensorflow에서 padding을 통해 shape 맞추기
        self.multi_labels.append(labels)#각각의 labels들이 묶음이 되도록 만들기


    print("multi_labels 개수 : ", len(self.multi_labels))
    print("정렬 전 : ",self.multi_labels)
      #단어들을 전체 문장들에서 나타난 빈도수에 맞춰 각 리스트 원소 정렬시키기!

        # 모든 단어를 추출하고 빈도수를 계산합니다.
    word_counts = Counter(word for sentence in self.multi_labels for word in sentence)
    print(type(word_counts))
        # 각 리스트를 단어 빈도수에 따라 정렬합니다. -> 각 리스트의 마지막 원소에 "end" 넣기
    self.multi_labels = [sorted(sentence, key=lambda word: word_counts[word], reverse=True) for sentence in self.multi_labels]
    for sentence in self.multi_labels:
      sentence.append("end")
        # 결과 출력
    for sentence in self.multi_labels:
      print(sentence)
        # 빈도수를 봄을로써 제대로 정렬되었는지 확인
    for word, count in word_counts.items():
      print(f'{word}: {count}')
    print("images 개수 : ", len(self.images))


  def image_preprocessing(self,input_images,size=(224,224)):#resize -> 뒤에서 쓸 CNN이 pretrained인 경우, padding으로 최대 크기에 맞춰야 하는 것이 아니라, resize로 CNN 입력값 크기에 맞춰야 한다! VGG에선 224*224
    import tensorflow as tf
    # 입력 이미지 생성 (예: 300x400 크기의 이미지)
    images=list([])
    for image in input_images:
      image = np.expand_dims(image, axis=0)#개수(batch)에 대한 차원이 없으니, 차원 추가
      image = tf.constant(image, dtype=tf.float32)
    # 이미지 크기 조절
      image = tf.image.resize(image, size, method='bilinear')#이 input_image 텐서의 shape은 [batch, height, width, channels] 형식이어야 한다.
      #method: 크기 조절 방법을 지정하는 문자열 매개변수입니다. 주로 사용되는 값은 "bilinear"와 "nearest"입니다. "bilinear"은 양선형 보간을 사용하여 부드럽게 크기를 조절하고, "nearest"는 가장 가까운 픽셀 값을 사용하여 크기를 조절
      image = np.squeeze(image, axis=0)
      images.append(image)
    images=np.array(images)
    return images