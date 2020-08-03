import os
import numpy as np
np.random.seed(1337)
import read_data
import mymodel
from keras.utils import np_utils,multi_gpu_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from sklearn.metrics import confusion_matrix

from keras.models import load_model

from keras.optimizers import Adam,RMSprop,Adamax
import matplotlib.pyplot as plt

# load data
# 16,196
X_s=read_data.read_dataset(r'E:\GYK\google_tts\TIMIT_split_1s')

# 14,663
X_c=read_data.read_dataset(r'E:\GYK\google_tts\TIMIT_wavnet_split_low2')
X=np.vstack((X_s,X_c))
y=read_data.read_label(r'.\label.txt')

# data preprocess
X_train, X_test, y_train, y_test_1 = train_test_split(X, y, test_size = 0.2, random_state= 0)

X_train = X_train.reshape(-1, X.shape[1], 1)
X_test = X_test.reshape(-1, X.shape[1], 1)
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test_1, num_classes=2)
print('...',y_test_1)
print('...', X_train.shape)
# Build model
# model=S_ResNet.s_res()
#model = mymodel.model_1(X)
# model=load_model('model.h5')
model = mymodel.origin(X)
#model=multi_gpu_model(model,gpus=4)

# Active model
adam=Adam(lr=1e-4)
# rmsprop=RMSprop(lr=1e-4)

model.compile(
    optimizer=adam,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train model
print("Training----------------------------------")
checkpoint=ModelCheckpoint('./Checkpoint/best.h5', monitor='val_loss', save_best_only=True, mode='auto',period=5)
early_stopping=EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=0)
lr_reduce=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=2,mode='auto',verbose=0)
history = model.fit(X_train,y_train,epochs=1, batch_size=64, verbose=1,validation_data=(X_test, y_test),
                    callbacks=[lr_reduce, early_stopping, checkpoint])
# history = model.fit(X_train,y_train,epochs=5,verbose=1,validation_data=(X_test, y_test))
history_dict = history.history

# test model
print("\nTesting------------------------------------")
loss,accuracy=model.evaluate(X_test,y_test)

print('test loss:',loss)
print('test accuracy:',accuracy)

y_pred = model.predict(X_test)
y_pred = [ 0 if y[0]>0.5 else 1 for y in y_pred]
y_pred=np.array(y_pred,dtype=float)
y_test_1=np.array(y_test_1,dtype=float)
print(y_pred.dtype)
print(y_pred.dtype)
# error = y_pred - y_test_1
# print('正确率为：{}'.format((1-sum(abs(e) for e in error)/len(error))))
print(confusion_matrix(y_test_1,y_pred))

# model.save('ROC_test.h5')

# figure acc
acc=history_dict['acc']
val_acc=history_dict['val_acc']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'r',linewidth=5.0,label='Training acc')
plt.plot(epochs,val_acc,'b',linewidth=5.0,label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# figure loss
loss_value=history_dict['loss']
val_loss_value=history_dict['val_loss']

plt.plot(epochs,loss_value,'r',linewidth=5.0,label='Training loss')
plt.plot(epochs,val_loss_value,'b',linewidth=5.0,label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# model.save('save_model/result_1.h5')
