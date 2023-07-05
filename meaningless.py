import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten
def model_base_64x3_CNN(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
    #
    model.add(Flatten())
    # 第四步，model.compile()
    model.compile(loss="mse", optimizer="Adam", metrics=["accuracy"])
    return model

if __name__ == '__main__':
    # input = np.random.rand(1, 300, 400,  3)
    # input = np.random.rand(1, 3, 300, 400)
    # net = model_base_64x3_CNN([300, 400, 3])
    # net.summary()
    # print(net.predict(input).shape)
    a = 0.99975
    a = pow(a, 10)
    print(a)
    a = pow(a, 10)
    print(a)
    a = pow(a, 10)
    print(a)
    a = pow(a, 10)
    print(a)
    a = pow(a, 2)
    print(a)


