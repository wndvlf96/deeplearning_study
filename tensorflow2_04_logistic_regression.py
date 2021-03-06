# 분류(Classificarion): 
# 미지의 입력 데이타가 무엇인지 예측(숫자가 아닌것)
# 데이터를 n차원에 두고 어떤 선(면)으로 나눌 수 있는 것을 찾기
# 그것을 바탕으로 예측하기
# 시그모이드 함수를 통해서 출력값을 1 또는 0으로만 출력
# 0.5 이상이면 1나올 확률이 높고, 반대면 반대

# Regression (Wx+b) 찾고
# Classification (sigmoid) 를 통해서 분류예측

# 손실함수
# y = 1/(1 + e^-(Wx+b))
# p(C=1 | x) = y = sigmoid(Wx+b) (1이 나올 확률)
# p(C=0 | x) = 1 - p(C=1 | x)
# p(C = t|x) = (y^t) * (1-y)^1-t
# 우도함수: 다수의 입력 x에 대해 정답 t가 발생될 확률
# 확률이 독립적이므로 각 입력 데이터 곱해서 우도함수 나타냄
# L(W,b) = 곱하기(i=1~n) p(C=ti | xi)
# 이 우도함수를 최소로하는 W,b를 구하기

# 케라스의 출력층에서 y를 계산해서 손실함수가 최소값인지 확인하기
