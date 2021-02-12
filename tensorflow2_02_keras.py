# TF 2.0에서 케라스를 사용할 것 권장
# 직관적인 API를 이용하면 ANN,CNN,RNN 쉽게 구축 가능

# Keras에서는 모델이 엄청 중요!
# 모델: 입력층(Flatten 가끔 Dense),
# 모델: 은닉층(Dense(2, activation='..)),
# 모델: 출력층(Dense)
# model = Sequential()로 모델 만듬
# model.add() 층 추가
# model.compile() 손실함수, 옵티마이저 지정
# model.fit() 학습 진행
# (손실함수 값이 최소가 될 때 까지
# 가중치와 바이어스를 업데이트하는 과정)
# model.evaluate() 평가
# model.save("qwe.h5") 학습 없이 다음에 이용가능 하게
# model = tf.keras.models.load_model("qwe.h5")


# 1. 데이터 생성
# 2. 모델 구축 model.Sequential(), model.add()
# 3. 모델 컴파일 model.compile()
# 4. 모델 학습 model.fit()
# 5. 모델 평가 및 예측 model.evaluate(), model.predict()
# 6. 모델 저장 model.save()

# train data: 학습 도중 사용, 가중치와 바이어스 최적화 위해 사용
# validation data: 학습 도중 사용, 1epoch마다 오버비팅 확인하기 위해 사용
# test data: 학습 후 정확도 평가, 임의의 입력에 대한 결과 예측을 위해 사용

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

# 데이터 가져오기(아마 +2 해주는거)
x_data = np.array([1,2,3,4,5,6])
y_data = np.array([3,4,5,6,7,8])

# 모델 구축
# 모델 생성
model = Sequential()
# 입력층
model.add(Flatten(input_shape=(1, )))
# 출력층
model.add(Dense(1, activation = 'linear'))
# 모델컴파일
model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')
# 모델 어떻게 생겼는지(각 층) 미리 확인해보기
model.summary()
# 모델 학습(여기서 epochs만큼 출력된다!)
hist = model.fit(x_data, y_data, epochs=1000)
# 모델 평가
result = model.predict(np.array([-3.1, 3.0, 3.5, 15.0, 20.1]))
print(result)