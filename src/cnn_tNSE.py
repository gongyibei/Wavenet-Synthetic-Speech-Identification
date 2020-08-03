import os
import numpy as np
np.random.seed(1337)
import read_data
import label
import mymodel
from keras.utils import np_utils,multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.optimizers import Adam,RMSprop,Adamax
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

model = load_model(r'E:\GYK\google_tts\save_model\train_bonafid_split_1s_vs_train_SS_1_split_1s')
print(model.layers[-2].name)
#dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('GlobalAveragePooling1D').output)
#dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
