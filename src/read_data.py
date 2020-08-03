import os
import numpy as np
np.random.seed(1337)
import scipy.io.wavfile as wave
from python_speech_features import mfcc
import python_speech_features
# from bispectrumd import bispectrumd
def read_dataset(filepath):
    data = []
    # h = [-1,2,-1]
    #h = [1,-1,0,0,0]
    h = [1,-2,1,0,0]
    # h = [1,-3,3,-1,0]
    # h = [1,-4,6,-4,1]
    for fn in os.listdir(filepath):
        if fn.endswith('.wav'):
            fd1 = os.path.join(filepath, fn)
            r1, x1 = wave.read(fd1)
            da=np.convolve(x1, h, 'same')
            # m1 = mdct.mdct(x1)
            data.append(da)
    X = np.array(data)
    return X

def read_label(filename):
    y = np.loadtxt(filename)
    return y


def read_dataset_mfc(filepath):
    data = []
    for fn in os.listdir(filepath):
        if fn.endswith('.wav'):
            fd1 = os.path.join(filepath, fn)
            fs, wav = wave.read(fd1)
            # da=np.convolve(x1, h, 'same')
            da = mfcc(wav,nfilt=41,numcep=40)
            np.array(da)
            da = [sum(i)/len(i) for i in da.T]
            # da = [da[i+1]-da[i] for i in range(len(da)-1)]
            data.append(da)

    X = np.array(data)
    print(X.shape)
    return X

def read_dcase_dataset_mfc(filepath):
    data = []
    for fn in os.listdir(filepath):
        if fn.endswith('.wav'):
            fd1 = os.path.join(filepath, fn)
            fs, wav = wave.read(fd1)
            # da=np.convolve(x1, h, 'same')
            data = python_speech_features.mfcc(wav,
                                               samplerate=48000,
                                               winlen=0.040,
                                               winstep=0.040,
                                               nfilt=40,
                                               numcep=40)
            np.array(da)
            # da = [sum(i)/len(i) for i in da.T]
            # da = [da[i+1]-da[i] for i in range(len(da)-1)]
            data.append(da)

    X = np.array(data)
    print(X.shape)
    return X
def Normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]

# def read_dataset_haso(filepath):
#     data = []
#     for fn in os.listdir(filepath):
#         if fn.endswith('.wav'):
#             fd1 = os.path.join(filepath, fn)
#             fs, wav = wave.read(fd1)
#             # da=np.convolve(x1, h, 'same')
#             da,_ = bispectrumd(wav ,128,3,64,0)
#             da = np.diag(da)
#             da = np.abs(da)
#             da = Normalize(da)
#             data.append(da)
#     X = np.array(data)
#     return X

if __name__ == '__main__':
    read_dataset_mfc(r'D:\龚永康\WaveNet\data\test')