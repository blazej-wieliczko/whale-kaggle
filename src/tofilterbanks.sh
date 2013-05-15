cd ../data/full-data/train
prefix=train
for i in $(ls wavpcm/ | tr -d 'abcdefghijklmnopqrstuvwxyz.')
do

sfbank -D --length 100 --shift 50 --freq-min 50 --freq-max 400 -f 2000 -F pcm16 -n 6 wavpcm/$prefix$i.wavpcm ./fbank/$prefix$i.fbank; 

echo "processed $prefix$i"

done

cd ../test
prefix=test
for i in $(ls wavpcm/ | tr -d 'abcdefghijklmnopqrstuvwxyz.')
do

sfbank -D --length 100 --shift 50 --freq-min 50 --freq-max 400 -f 2000 -F pcm16 -n 6 ./wavpcm/$prefix$i.wavpcm ./fbank/$prefix$i.fbank; 

echo "processed $prefix$i"

done
