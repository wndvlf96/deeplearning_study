import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
print(tf.__version__)

# 데이터 로드
(x_train, t_train),(x_test, t_test) = fashion_mnist.load_data()

# 데이터 보기
# (형태, 이미지라면 흑백 혹은 rgb 확인하기 위해서)

# 5행 5열의 칸 생성
plt.figure(figsize=(6,6))
# 25개만 보기
for i in range(25):

    # 한 그래프에서 여러개의 그래프를 그릴 때 사용 subplot()
    # 매개변수로 총 행수, 총 열수, 격자의 인덱스 들어온다!!!
    plt.subplot(5, 5, i+1)

    # cmap = 'gray'로 거의 적외선 급의 사진을 흑백사진으로 교체
    plt.imshow(x_train[i], cmap='gray')
    
    # 축 수치 표현하지 않기
    # 이걸로 height, width 확인하기 어차피 어려움!!!
    # x_train, t_train, x_test, t_test 모두 다 .shape를 찍어서 직접 확인하자!
    plt.axis('off')
plt.show()

# 데이터 전처리

# 흑백 사진이므로 채널 하나로 지정해줌!
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))

print("train x shape: ", x_train.shape)
print("train label shape: ", t_train.shape)
print("test x shape: ", x_test.shape)
print('test label shape: ', t_test.shape)

# 픽셀(0 ~ 255) 데이터 0 ~ 1사이로 맞추기
x_train= (x_train - 0.0)/(255.0 - 0.0)
x_test = (x_test - 0.0)/(255.0 - 0.0)

# 원핫 인코딩은 안 할 것이므로 이따가 loss = 'sparse_categorical_crossentropy' 사용하기

# 모델 구축

model = tf.keras.Sequential()
# 컨볼루션층 input_shape train.shape과 똑같이 맞춰준다.
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
# padding이 없으므로 width, height 줄어든다. 그에 따라 커널의 개수를 늘려서 그만큼 더 많이 실행.
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

# 완전연결층
model.add(tf.keras.layers.Flatten())

# 출력층
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# 모델 학습
hist = model.fit(x_train, t_train, epochs=5, validation_split=0.3)
# 모델 평가
model.evaluate(x_test,t_test)