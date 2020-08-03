import scipy.io.wavfile as wave
from bispectrumd import bispectrumd
import os
for file os.listdir(r'E:\GYK\google_tts\TIMIT_split_1s_200'):

fs, wav = wave.read(r'E:\GYK\google_tts\TIMIT_split_1s_200\22_1.wav')
da, _ = bispectrumd(wav, 128, 3, 64, 0)

fs, wav = wave.read(r'E:\GYK\google_tts\TIMIT_wavnet_split_low2_200\en-AU-Wavenet-A_0_1.2_sx118_1.wav')
da, _ = bispectrumd(wav, 128, 3, 64, 0)