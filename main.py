import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print('shape of x_train:', x_train.shape)
print('shape of y_train:', y_train.shape)
print('shape of x_test:', x_test.shape)
print('shape of y_test:', y_test.shape)

# Mnist 데이터셋이 어떻게 생겼는지 확인 ##
plt.rcParams['figure.figsize'] = (5, 5)
plt.imshow(x_train[0])
plt.show()


# reshape and normalization
# 모델의 input으로 넣기 위해 (28 * 28 = 784) 데이터 형태로 변형, 0~1 사이의 실수 값으로 정규화
x_train = x_train.reshape((60000, 28 * 28)) / 255.0
x_test = x_test.reshape((10000, 28 * 28)) / 255.0

# Sequential model
model = tf.keras.models.Sequential()

# Stacking layers
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(28*28,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()



#비용함수 및 최적화 함수 설정
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=1,
          verbose=1,
          validation_split=0.2)



#학습된 모델 성능 평가 테스트
test_loss, test_acc = model.evaluate(x_test, y_test)

print('\n테스트 정확도 : ', test_acc)
predictions = model.predict(x_test)
print((np.argmax(predictions[0])))
print(y_test[0])
