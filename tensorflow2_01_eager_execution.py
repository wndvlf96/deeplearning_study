# Eager Execution
# 즉시 실행 모드

import tensorflow as tf
import numpy as np

# successfully dll open 하려면 -> 쿠다 적절한거 다운 + VC_redist.x64.exe
tf.config.list_physical_devices('GPU')

print("--------------------------")
print("텐서플로우 버전: "+str(tf.__version__))
print("--------------------------")

a = tf.constant(1.0)
b = tf.constant(2.0)
# 텐서플로우 1.x에서는 a+b를 출력하면 Tensor라는 구문이 나ㅁ옴
# 그러나 2.x에서는 세션 필요없이 3나오게 가능
c = a+b
print("--------------------------")
print("c의 값은: "+str(c.numpy()))
print("--------------------------")

# 바로 변수 초기화 가능!
W = tf.Variable(tf.random.normal([1]))
print('초기 W= ', W.numpy())

for step in range(2):
    W = W+1.0
    # numpy를 사용하면 바로 출력도 가능
    print('step= ',step,' , W = ', W.numpy())

# placeholder 삭제 
# (Lazy Evaluation) 실제 데이터는 세션내에서 입력받으려고