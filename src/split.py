# -*- coding: utf-8 -*-

import wave
import os

def audio_split(file_name, path):
    """split

    :param file_name:需要裁剪的音频文件名
    :param path:裁剪音频存放路径
    """
    with wave.open(file_name, 'rb') as f:
        framerate = f.getframerate()
        nframs = f.getnframes()
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()

        i = 1
        while nframs >= int(framerate*1.5):
            with wave.open('{}/{}_{}.wav'.format(path, os.path.basename(file_name)[:-4], i),
                           'wb') as fo:
                fo.setnchannels(nchannels)
                fo.setsampwidth(sampwidth)
                fo.setframerate(framerate)
                fo.writeframes(f.readframes(int(framerate*1.5)))
            nframs -= int(framerate*1.5)
            i += 1

for root, dirs, files in os.walk(r'E:\Lyz\TIMIT'):
    for name in files:
        if name.endswith('wav'):
            audio_split(os.path.join(root,name), 'E:\GYK\google_tts\TIMIT_split_1_5s')

