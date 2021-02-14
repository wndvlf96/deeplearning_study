# Limitation of explicit programming
# (일일이 프로그래밍 하기에는 너무 규칙이 많아서)
# 그래서 직접 데이터를 학습해서 예측을 하게하자!!!

# Supervised Learning
# 학습하는 방법에 따라서 un이 붙을지 아닐지 나뉨
# 데이터에 따른 label이 달려있는 것으로 학습

# Regression: label이 점수(연속적)
# Binaray classification: 분류(라벨)가 2개일 때
# Multi-label classification: 분류가 여러 개 이상일 때

# Unsupervised Learning
# 데이터에 따른 label이 미리 안 달려있는 상태로 학습

import tensorflow as tf
import numpy as np

print(tf.__version__)

# 텐서플로우2.x 는 session이 없이도 출력가능!
# 플레이스홀더도 없어졌음!
print('------------------------')
hello = tf.constant('hello world')
print(hello)
n1 = tf.constant(3.0, tf.float32)
n2 = tf.constant(5.0, tf.float32)
print(n1+n2)

# rank: 몇 차원인지 나타냄 (0은 스칼라, 1은 1차원 배열에 값)
# shape: 형태
# type: 보통 tf.float32

# supervised learning중 Regression
print('------------------------')

# 데이터 준비
x_train = np.array([[10], [9], [3], [21],[46],[46],[12],[12],[35],[46],[15],[34],[58],[13],[84],[26],[48],[13],[54]])
t_train = np.array([[10], [9], [3], [21],[46],[46],[12],[12],[35],[46],[15],[34],[58],[13],[84],[26],[48],[13],[54]])

# 전처리? 필요없을 듯
# 모델 구성
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape = (1, ), activation='linear'))
# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5), loss= 'mse')
model.summary()
# 모델 학습
hist = model.fit(x_train, t_train, epochs=100)
# 모델 평가
x_test = [14,25,36,17,25]
t_test = [14,25,36,17,25]
pred_val = model.predict(np.array(x_test))
print(pred_val)
print(t_test)
# 모델 입력
print(model.input)
# 모델 출력
print(model.output)
# 모델 가중치
print(model.weights)