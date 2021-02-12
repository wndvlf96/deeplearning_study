# Regression(회귀)
# 훈련 데이터로 그 데이터가 아닌 데이터를 연속적 (숫자)값으로 예측하는 것

# 학습 개념
# 트레이닝 함수로 (일차)함수를 만들어 내는 것
# 일차함수에서 가중치(W), 바이어스(b (y절편)) 찾기

# 오차 = t - (Wx + b)
# 오차가 크면 W,b 잘못 설정한 것
# 머신러닝은 이 오차를 최소화하는 것

# 손실함수
# 정답(t)와 입력(x)에 대한 y값의 차이를 모두 더해 수식으로 더한 것
# 오차계산시 제곱으로 함(음수배제, 제곱이여서 수 더 커짐)
# 즉 손실함수는 모든 오차값에 대한 평균값
# 이것은 W와 b에만 영향을 받음!

# 경사하강법(Gradient Descent alg.)
# 손실함수가 포물선으로 그려짐
# 기울기가 더이상 작아지지 않는 부분 찾는 방법
# 1. 임의의 W선택
# 2. 그 W에서의 미분값과 편비분값 구하고
# 3. 그값이 작아지는 방향으로 W감소(또는 증가)
# 4. 최종적으로 기울기가 더 이상 작아지지 않는 부분 찾기
# 편미분 값이 양수이면 왼쪽, 양수이면 오른쪽

# 입력값 3개와 출력 있을 때 이것에 대한 수식 만들기
# 케라스의 출력층에서 계산

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np

# 데이터 생성
x_data = np.array([[1,2,0],[5,4,3],
    [1,2,-1],[3,1,0],[2,4,2],[4,1,2],
    [-1,3,2],[4,3,3],[0,2,6],[2,2,1],
    [1,-2,-2],[0,1,3],[1,1,3],[0,1,4],[2,3,3]])
t_data = np.array([-4, 4, -6, 3, -4, 9, -7, 5, 6, 0,4, 3,5,5,1])
print('x:',x_data.shape,' , y:', t_data.shape)

# 모델 구축
model = Sequential()
# 입력 3개여서 shape=(3, ) 선형회귀이므로 linear
model.add(Dense(1, input_shape=(3, ), activation='linear'))

# 모델 컴파일
# 학습알고리즘: SGD, 선형회귀여서 mse
model.compile(optimizer=SGD(learning_rate=1e-2),loss='mse')
model.summary()

# 모델 학습
hist = model.fit(x_data, t_data, epochs=1000)

# 모델 평가
test_data=[[5,5,0],[2,3,1],[-1,0,-1],[10,5,2],[4,-1,-2]]
ret_val=[2*data[0] -3 * data[1] + 2*data[0] for data in test_data]
pred_val = model.predict(np.array(test_data))
print(pred_val)
print(ret_val)
# 모델 입력
print(model.input)
# 모델 출력
print(model.output)
# 모델 가중치
print(model.weights)