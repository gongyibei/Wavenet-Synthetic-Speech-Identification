import os
import numpy as np
np.random.seed(1337)
import read_data
import label
import mymodel
import sklearn.metrics as sm
from keras.utils import np_utils,multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.optimizers import Adam,RMSprop,Adamax
import matplotlib.pyplot as plt



isSplit = False
# load data
# 16,196
# T = "all_bonafid_split_1s"
T = "TIMIT_split_1s"
X_s=read_data.read_dataset(r'D:\GYK\WaveNet\data\{}'.format(T))

# 14,663
F = "TIMIT_WavNet_split_1s"
X_c=read_data.read_dataset(r'D:\GYK\WaveNet\data\{}'.format(F))
X=np.vstack((X_s,X_c))


# creat label
m = X_s.shape[0]
n = X_c.shape[0]
label.creat_label(m,n)
y=read_data.read_label(r'.\label.txt')



# data preprocess
if isSplit:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)
else :
    X_train = X
    y_train = y
    t = "TIMIT_split_1s"
    X_s = read_data.read_dataset(r'D:\GYK\WaveNet\data\{}'.format(t))

    # 14,663
    f = "TIMIT_wavnet_split_low2"
    X_c = read_data.read_dataset(r'D:\GYK\WaveNet\data\{}'.format(f))
    X_test = np.vstack((X_s, X_c))

    # creat label
    m = X_s.shape[0]
    n = X_c.shape[0]
    label.creat_label(m, n)
    y_test = read_data.read_label(r'.\label.txt')


X_train = X_train.reshape(-1, X.shape[1], 1)
X_test = X_test.reshape(-1, X.shape[1], 1)
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)


# Build model
if 0 :

    model=load_model('model.h5')
else:
    #model=S_ResNet.s_res()
    #model = mymodel.model_1(X)
    #model=multi_gpu_model(model,gpus=4)
    model = mymodel.origin(X)

# Active model
adam=Adam(lr=1e-4)
# rmsprop=RMSprop(lr=1e-4)

model.compile(
    optimizer= adam,
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# train model
print("Training----------------------------------")
checkpoint=ModelCheckpoint('./Checkpoint/best.h5', monitor='val_loss', save_best_only=True, mode='auto',period=5)
early_stopping=EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=0)
lr_reduce=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,mode='auto',verbose=0)
history = model.fit(X_train,y_train,epochs=10, batch_size=64, verbose=1,validation_data=(X_test, y_test),
                    callbacks=[lr_reduce, early_stopping, checkpoint])
# history = model.fit(X_train,y_train,epochs=10,verbose=1,validation_data=(X_test, y_test))
history_dict = history.history

# test model
print("\nTesting------------------------------------")
loss,accuracy=model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)

# y_pred = model.predict(X_test)
# error = y_pred - y_test
# print('正确率为：{}'.format((1-sum(abs(e) for e in error)/len(error))))
# print(confusion_matrix(y_test,y_pred))

model.save(r'D:\GYK\google_tts\save_model\{}_all_vs_{}_all'.format(T,F))

# figure acc
# acc=history_dict['acc']
# val_acc=history_dict['val_acc']
# epochs=range(1,len(acc)+1)

# plt.plot(epochs,acc,'r',linewidth=5.0,label='Training acc')
# plt.plot(epochs,val_acc,'b',linewidth=5.0,label='Validation acc')
# plt.title('Training and validation acc')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#plt.show()
#plt.savefig(r'D:\GYK\google_tts\save_model\acc_{}_vs_{}'.format(T,F))

# figure loss
# loss_value=history_dict['loss']
# val_loss_value=history_dict['val_loss']

# plt.plot(epochs,loss_value,'r',linewidth=5.0,label='Training loss')
# plt.plot(epochs,val_loss_value,'b',linewidth=5.0,label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# plt.savefig(r'D:\GYK\google_tts\save_model\loss_{}_vs_{}'.format(T,F))

# model.save('save_model/result_1.h5')
