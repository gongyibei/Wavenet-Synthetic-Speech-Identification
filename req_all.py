# -*- coding: utf-8 -*-

import os
import json
import base64
import pprint

import numpy as np

URL = 'https://cxl-services.appspot.com/proxy?url=https%3A%2F%2Ftexttospeech.googleapis.com%2Fv1beta1%2Ftext%3Asynthesize'

def mk_audiodir(file):
    if not os.path.exists(file):
        os.mkdir(file)

def write(audio, file):
    """write
    生成mp3文件
    :param audio:音频数据
    :param file:音频存放路径
    """
    if not os.path.exists(os.path.dirname(file)):
        os.mkdir(os.path.dirname(file))
    with open(file, 'wb') as f:
        f.write(audio)


def gen_cmd(**kw):
    """gen_cmd
    根据参数生成命令

    :param **kw:获取参数
    """
    audioEncoding = kw.get('audioEncoding', 'LINEAR16')
    pitch = kw.get('pitch', '0.00')
    speakingRate = kw.get('speakingRate', '1.00')
    text = kw.get('text', "Hello World")
    name = kw.get('name', 'en-US-Wavenet-D')
    languageCode = name[:5]

    url = URL
    data = {
        "audioConfig": {
            "audioEncoding": audioEncoding,
            "pitch": pitch,
            "speakingRate": speakingRate
        },
        "input": {
            "text": text
        },
        "voice": {
            "languageCode": languageCode,
            "name": name
        }
    }
    cmd = 'curl --socks5 127.0.0.1:1080 -H "content-type: text/plain;charset=UTF-8" -H "Origin: http://dkcldgdkd.jumpbc.chuairan.com" -H "Referer: http://dkcldgdkd.jumpbc.chuairan.com/" -d "{0}" "{1}"'.format(
        data, url)
    return cmd


def get_audio(cmd):
    audio_json_str = os.popen(cmd)
    audio_json = json.load(audio_json_str)
    audioContent = audio_json['audioContent']
    audio = base64.b64decode(audioContent.encode('utf8'))
    return audio


def get_texts(file='./PROMPTS.TXT'):
    with open(file) as f:
        lines = f.readlines()
    lines = lines[6:]
    return lines

def main():
    mk_audiodir('./audio_all/')
    texts = get_texts()
    names = [
        "en-US-Wavenet-A", "en-US-Wavenet-B", "en-US-Wavenet-C",
        "en-US-Wavenet-D", "en-US-Wavenet-E", "en-US-Wavenet-F",
        "en-AU-Wavenet-A", "en-AU-Wavenet-B", "en-AU-Wavenet-C",
        "en-AU-Wavenet-D", "en-GB-Wavenet-A", "en-GB-Wavenet-B",
        "en-GB-Wavenet-C", "en-GB-Wavenet-D"
    ]
    pitchs = list(map(lambda x:str(x)+'.00',np.arange(-8,9,4))) # -20~20 step:1
    speakingRates = list(map(lambda x:str(x),np.arange(0.8,1.3,0.2))) # 0~4 step:0.25
    for name in names:
        mk_audiodir('./audio_all/{0}'.format(name))
        for pitch in pitchs:
            for speakingRate in speakingRates:
                mk_audiodir('./audio_all/{0}/{1}_{2}'.format(name,pitch,speakingRate))
                for text_with_id in texts:
                    text,text_id = text_with_id.strip().split(' (')
                    text = text.replace("'","’")
                    cmd = gen_cmd(text=text, name=name, pitch=pitch, speakingRate=speakingRate)
                    audio = get_audio(cmd)
                    write(audio, './audio_all/{0}/{1}_{2}/{3}.mp3'.format(
                        name, pitch,speakingRate, text_id[:-1]))


if __name__ == '__main__':
    main()
    #get_texts()
