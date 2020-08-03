import numpy as np
np.random.seed(1337)
from keras.utils import plot_model
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Conv1D,MaxPooling1D,GlobalAveragePooling1D,Activation,AveragePooling1D,Lambda,Dropout
from keras.layers.advanced_activations import ThresholdedReLU,PReLU
from keras.initializers import RandomNormal
# from Blocks import Group,pre_block,res_block1,res_block2
import keras.backend as K
# def fix(x):
#     h=[-1,2,-1]
#     y=K.conv1d(x,h,padding='same')
#     return y
# def sub_mean(x):
#     x=-K.mean(x,axis=1,keepdims=True)

init = RandomNormal(mean=0,stddev=0.001)

def origin(X):
    model=Sequential()
    # 1st layer
    model.add(Conv1D(
        input_shape=(X.shape[1], 1),
        filters=1,
        kernel_size=5,
        padding='same'
    ))

    model.add(Conv1D(8, 1, padding='same'))
    model.add(Conv1D(8, 3, strides=2, padding='same'))

    # 2nd layer
    model.add(Conv1D(8, 5, padding='same'))
    model.add(Conv1D(16, 1, padding='same'))
    model.add(Conv1D(16, 3, strides=2, padding='same'))

    # 3rd layer
    model.add(Conv1D(16, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv1D(32, 1, padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    # 4th layer
    model.add(Conv1D(32, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv1D(64, 1, padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    # 5th layer
    model.add(Conv1D(64, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv1D(128, 1, padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    # 6th layer
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv1D(256, 1, padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))

    # 7th layer
    model.add(Conv1D(256, 5, padding='same'))
    model.add(Activation('tanh'))
    model.add(Conv1D(512, 1, padding='same'))
    model.add(Activation('tanh'))
    model.add(GlobalAveragePooling1D(name='pooling'))

    #model.add(Dropout(0.5))

    # 8th layer(FC & softmax)
    # model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.summary()

    return model

def model_1(X):
    model = Sequential()

    # 1st layer
    model.add(Conv1D(
        input_shape=(X.shape[1], 1),
        filters=1,
        kernel_size=5,
        padding='same'
    ))
    model.add(Conv1D(8, 1, padding='same'))
    # model.add(Lambda(abs))
    model.add(ThresholdedReLU(theta=1.0))
    # model.add(Conv1D(8,3,strides=2,padding='same'))

    # 2nd layer
    model.add(Conv1D(8, 5, padding='same'))
    model.add(Conv1D(16, 1, padding='same'))
    model.add(Activation('relu'))
    # model.add(PReLU())
    # model.add(Conv1D(16,3,strides=2,padding='same'))

    # 3rd layer
    model.add(Conv1D(16, 5, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(Conv1D(32, 1, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=3, strides=2, padding='same'))
    # model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))

    # 4th layer
    model.add(Conv1D(32, 5, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(Conv1D(64, 1, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=3, strides=2, padding='same'))
    # model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))

    # 5th layer
    model.add(Conv1D(64, 5, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(Conv1D(128, 1, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=3, strides=2, padding='same'))
    # model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))

    # 6th layer
    model.add(Conv1D(128, 5, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(Conv1D(256, 1, padding='same'))
    # model.add(PReLU())
    model.add(Activation('relu'))
    model.add(AveragePooling1D(pool_size=3, strides=2, padding='same'))
    # model.add(MaxPooling1D(pool_size=3,strides=2,padding='same'))

    # 7th layer
    model.add(Conv1D(256, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(512, 1, padding='same'))
    model.add(Activation('relu'))

    # model.add(LSTM(50,return_sequences=False))
    model.add(GlobalAveragePooling1D())

    # model.add(LSTM(50,return_sequences=False))
    # # model.add(Dense(128))
    #
    model.add(Dropout(0.25))

    # model.add(Dense(128))

    # 8th layer(FC & softmax)
    model.add(Dense(2, activation='softmax'))

    model.summary()

    return model

def model_block(X):
    inputs = Input(shape=(X.shape[1],1))

    x = Conv1D(1,5,padding='same')(inputs)
    x = Conv1D(8,1,padding='same')(x)
    x = ThresholdedReLU(theta=1.0)(x)

    x = Conv1D(8,5,padding='same')(x)
    x = Conv1D(16, 1, padding='same')(x)

    group1 = Group(filters=16,x=x)
    group2 = Group(filters=32,x=group1)
    group3 = Group(filters=64,x=group2)
    group4 = Group(filters=128,x=group3)
    group5 = Group(filters=256,x=group4)

    fc = GlobalAveragePooling1D()(group5)
    outputs = Dense(2,activation='softmax')(fc)

    model = Model(inputs,outputs)
    model.summary()

    return model

def res_model(X):
    inputs = Input(shape=(X.shape[1], 1))

    x = Conv1D(8, 5, padding='same')(inputs)

    x = res_block2(8, x)
    x = res_block2(8, x, pool=True)
    x = res_block2(16, x)
    x = res_block2(16, x, pool=True)
    x = res_block2(32, x)
    x = res_block2(32, x, pool=True)
    x = res_block2(64, x)
    x = res_block2(64, x, pool=True)
    x = res_block2(128, x)
    x = res_block2(128, x, pool=True)
    x = res_block2(256, x)
    x = res_block2(256, x, pool=True)


    fc = GlobalAveragePooling1D()(x)
    fc = Dense(128,activation='relu')(fc)
    fc = Dropout(0.25)(fc)
    outputs = Dense(2, activation='softmax')(fc)

    model = Model(inputs, outputs)
    model.summary()
    return model

