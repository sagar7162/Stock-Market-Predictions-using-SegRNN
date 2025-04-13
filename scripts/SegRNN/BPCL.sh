if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

root_path_name=./dataset/ #for --root_path
data_path_name=BPCL.csv #for --data_path
model_id_name=BPCL
data_name=custom #for --data
features=MS #for --features
target=Close
freq=d #for --freq
model_name=SegRNN


#SegRNN
rnn_type=rnn #for --rnn_type
dec_way=pmf #for --dec_way
seg_len=48 #for --seg_len
win_len=48 #for --win_len

seq_len=720
for pred_len in 96
do
    python -u run_longExp.py \
      --is_training 1 \
      --do_predict \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features $features \
      --target $target \
      --freq $freq \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 48 \
      --win_len $win_len \
      --enc_in 9 \
      --d_model 512 \
      --dropout 0.5 \
      --train_epochs 30 \
      --patience 1 \
      --rnn_type $rnn_type \
      --dec_way pmf \
      --channel_id 1 \
      --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

#"Date" nhi "date" rakhna h csv me 