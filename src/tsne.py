# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
import read_data
from sklearn import datasets
from sklearn.manifold import TSNE
import os
import numpy as np
#np.random.seed(1337)
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
import numpy as np
np.random.seed(1337)
from keras.utils import plot_model
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Conv1D,MaxPooling1D,GlobalAveragePooling1D,Activation,AveragePooling1D,Lambda,Dropout
from keras.layers.advanced_activations import ThresholdedReLU,PReLU
from keras.initializers import RandomNormal

def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def get_mfcc(data):
    label = []
    TIMIT = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('TIMIT_split_1s_200'))
    label += [1]*TIMIT.shape[0]
    T_Wavenet = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('TIMIT_wavnet_split_low2_200'))
    label += [2] * T_Wavenet.shape[0]
    Bonafid = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('train_bonafid_10'))
    label += [3] * Bonafid.shape[0]
    ASV_Wavenet = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('train_SS_1_10'))
    label += [4] * ASV_Wavenet.shape[0]

    data = np.vstack((TIMIT,T_Wavenet,Bonafid,ASV_Wavenet))
    return data,label
def get_cnn():
    model = load_model(r'E:\GYK\google_tts\save_model\all_bonafid_split_1s_all_vs_all_SS_1_split_1s_all')
    print(model.layers)
    #dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling1d_1').output)
    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('pooling').output)
    n = 500
    label = []
    TIMIT = read_data.read_dataset(r'E:\GYK\google_tts\data\{}'.format('TIMIT_split_1s'))
    print(np.random.choice(range(TIMIT.shape[0]),500))
    print(TIMIT[1])
    TIMIT = np.array([TIMIT[i] for i in np.random.choice(range(TIMIT.shape[0]),500)])
    print(TIMIT.shape)
    TIMIT = TIMIT.reshape(-1, TIMIT.shape[1], 1)
    TIMIT = dense1_layer_model.predict(TIMIT)
    label += [1]*TIMIT.shape[0]
    T_Wavenet = read_data.read_dataset(r'E:\GYK\google_tts\data\{}'.format('TIMIT_wavnet_split_low2'))
    T_Wavenet = np.array([T_Wavenet[i] for i in np.random.choice(range(T_Wavenet.shape[0]), 500)])
    T_Wavenet = T_Wavenet.reshape(-1, T_Wavenet.shape[1], 1)
    T_Wavenet = dense1_layer_model.predict(T_Wavenet)
    label += [2] * T_Wavenet.shape[0]
    Bonafid = read_data.read_dataset(r'E:\GYK\google_tts\data\{}'.format('all_bonafid_split_1s'))
    Bonafid = np.array([Bonafid[i] for i in np.random.choice(range(Bonafid.shape[0]), 500)])
    Bonafid = Bonafid.reshape(-1, Bonafid.shape[1], 1)
    Bonafid = dense1_layer_model.predict(Bonafid)
    label += [3] * Bonafid.shape[0]
    ASV_Wavenet = read_data.read_dataset(r'E:\GYK\google_tts\data\{}'.format('all_SS_1_split_1s'))
    ASV_Wavenet = np.array([ASV_Wavenet[i] for i in np.random.choice(range(ASV_Wavenet.shape[0]), 500)])
    ASV_Wavenet = ASV_Wavenet.reshape(-1, ASV_Wavenet.shape[1], 1)
    ASV_Wavenet = dense1_layer_model.predict(ASV_Wavenet)
    label += [4] * ASV_Wavenet.shape[0]
    data = np.vstack((TIMIT, T_Wavenet, Bonafid, ASV_Wavenet))
    return data,label
def main():
    data,label = get_cnn()

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))

    plt.show(fig)


if __name__ == '__main__':
    main()