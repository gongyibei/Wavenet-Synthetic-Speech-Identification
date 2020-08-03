import read_data
import pandas as pd
from pandas import DataFrame,Series

import matplotlib.pyplot as plt

TIMIT = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('TIMIT_split_1s_200'))
T_Wavenet = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('TIMIT_wavnet_split_low2_200'))
Bonafid = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('train_bonafid_10'))
ASV_Wavenet = read_data.read_dataset_mfc(r'E:\GYK\google_tts\data\{}'.format('train_SS_1_10'))
for i in range(1,13):
    df = DataFrame({'TIMIT':TIMIT[:,i],'T_Wavenet':T_Wavenet[:,i],'Bonafid':Bonafid[:200,i],'ASV_Wavenet':ASV_Wavenet[:200,i]})
    boxplot=df.boxplot()
    plt.ylabel("ylabel")
    plt.xlabel("different datasets")
    plt.savefig('boxfig/{}'.format(i))
    plt.close()
