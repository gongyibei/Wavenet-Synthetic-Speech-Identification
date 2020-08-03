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

T = "all_bonafid_split_1s"
X_s=read_data.read_dataset(r'E:\GYK\google_tts\data\{}'.format(T))

# 14,663
F = "all_SS_1_split_1s"
X_c=read_data.read_dataset(r'E:\GYK\google_tts\data\{}'.format(F))

X_test=np.vstack((X_s,X_c))



m = X_s.shape[0]
n = X_c.shape[0]
y_test = [1]*m+[0]*n
# label.creat_label(m,n)
# y_test=read_data.read_label(r'.\label.txt')

X_test = X_test.reshape(-1, X_test.shape[1], 1)
#y_test = np_utils.to_categorical(y_test, num_classes=2)

model = load_model(r'E:\GYK\google_tts\save_model\TIMIT_split_1s_vs_TIMIT_wavnet_split_low2')


print("\nTesting------------------------------------")
#loss,accuracy=model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)
y_pred = [ 0 if t[0]>t[1] else 1 for t in y_pred]
#y_pred = np_utils.to_categorical(y_pred, num_classes=2)
cm = confusion_matrix(y_test,y_pred)
print(cm)
tpr = cm[0][0]/(cm[0][0]+cm[0][1])
print(tpr)
fpr = cm[1][1]/(cm[1][0]+cm[1][1])
print(fpr)

acc = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(acc)
# print('test loss:',loss)
# print('test accuracy:',accuracy)


