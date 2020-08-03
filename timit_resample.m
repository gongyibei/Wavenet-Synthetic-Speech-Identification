
maindir = '.\google_tts\TIMIT_wavnet_split_1s';
waves  = dir( maindir );
outdir = '.\google_tts\TIMIT_wavnet_split';

for i = 3 : length( waves )


    inpath = fullfile( maindir, waves( i ).name)
    outpath = fullfile( outdir, waves( i ).name)
    [data , fs] = audioread(inpath);
    data = resample(data,16000,24000);
    audiowrite(outpath,data,16000);

        % 此处添加你的对文件读写操作 %
end
