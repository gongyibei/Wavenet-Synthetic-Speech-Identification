
maindir = 'E:\GYK\google_tts\TIMIT_wavnet_split';
waves  = dir( maindir );
outdir = 'E:\GYK\google_tts\TIMIT_wavnet_split_low2';


% [data1 , fs] = audioread('E:\GYK\google_tts\TIMIT_split_1s\1_2.wav');
% [data2 , fs] = audioread('E:\GYK\google_tts\TIMIT_wavnet_split\en-AU-Wavenet-A_-1_0.8_sa1_2.wav');
% rate = max(abs(data1))/max(abs(data2))


for i = 3 : length( waves )


    inpath = fullfile( maindir, waves( i ).name);
    outpath = fullfile( outdir, waves( i ).name)
    [data , fs] = audioread(inpath);
    data = data/5;
    audiowrite(outpath,data,16000);

        % 此处添加你的对文件读写操作 %
end
