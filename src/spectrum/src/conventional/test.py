# -*- coding: utf-8 -*-

from bispectrumd import bispectrumd
import wave

def test():
    data = wave.Wave_read('/home/ukaii/project/WaveNet/Timit-WaveNet-audio/audio/en-AU-Wavenet-A/en-AU-Wavenet-A_0.00_1.00_sa1.wav') 
    data = data.readframes(data.getnframes())
    dbic = bispectrumd(data, 128,3,64,0)

test()
