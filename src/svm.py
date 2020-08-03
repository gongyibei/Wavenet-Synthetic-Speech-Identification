import os
import numpy as np
np.random.seed(1337)
import read_data
import label
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# load data
T = "TIMIT_split_1s"
X_s=read_data.read_dataset_mfc(r'D:\GYK\WaveNet\data\{}'.format(T))
print('read X_s done...')

F = "TIMIT_WavNet_split_1s"
X_c=read_data.read_dataset_mfc(r'D:\GYK\WaveNet\data\{}'.format(F))
print('read X_c done...')
X=np.vstack((X_s,X_c))

m = X_s.shape[0]
n = X_c.shape[0]
label.creat_label(m,n)
y=read_data.read_label(r'D:\GYK\WaveNet\cnn\label.txt')

# T = "all_bonafid_split_1s"
# X_s=read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format(T))
# print('read X_s done...')

# F = "all_SS_1_split_1s"
# X_c=read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format(F))
# print('read X_c done...')
# X_test=np.vstack((X_s,X_c))

# m = X_s.shape[0]
# n = X_c.shape[0]
# label.creat_label(m,n)
# y_test=read_data.read_label(r'E:\GYK\google_tts\cnn\label.txt')





# data preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)
# y_train = np_utils.to_categorical(y_train, num_classes=2)
# y_test = np_utils.to_categorical(y_test, num_classes=2)


clf = svm.SVC(kernel='linear', C=0.4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
error = y_pred - y_test
print('正确率为：{}'.format((1-sum(abs(e) for e in error)/len(error))))
print(confusion_matrix(y_test,y_pred))
