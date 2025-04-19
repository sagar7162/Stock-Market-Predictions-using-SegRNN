#!/bin/bash
# BPCL script that accepts command-line parameters

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Default parameter values
root_path_name=./dataset/ 
data_path_name=BPCL.csv
model_id_name=BPCL
data_name=custom
features=MS
target=Close
freq=d
model_name=SegRNN
rnn_type=rnn
dec_way=pmf
seg_len=48
win_len=48
seq_len=720
pred_len=96
is_training=1  # Default is training mode
skip_test=0    # Default is to run test phase

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --is_training) is_training="$2"; shift 2;;
    --skip_test) skip_test="$2"; shift 2;;
    --pred_len) pred_len="$2"; shift 2;;
    --seq_len) seq_len="$2"; shift 2;;
    --model) model_name="$2"; shift 2;;
    *) echo "Unknown parameter passed: $1"; exit 1;;
  esac
done

echo "Running with is_training=$is_training, skip_test=$skip_test, pred_len=$pred_len, seq_len=$seq_len"

# Create the model_id string properly (without single quotes)
full_model_id="${model_id_name}_${seq_len}_${pred_len}"

# After preprocessing, we have 4 input features and will predict 1 output feature (Close)
enc_in=5
dec_in=5
c_out=1

# Use the parameters in the command
python -u run_longExp.py \
  --is_training $is_training \
  --do_predict \
  --skip_test $skip_test \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $full_model_id \
  --model $model_name \
  --data $data_name \
  --features $features \
  --target $target \
  --freq $freq \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len $seg_len \
  --win_len $win_len \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 30 \
  --patience 1 \
  --rnn_type $rnn_type \
  --dec_way $dec_way \
  --channel_id 1 \
  --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

# If we're in test mode, display a helpful message
if [ "$is_training" -eq 0 ]; then
  echo "Processing completed. Results are available in: ./results/${full_model_id}_${model_name}_custom_ftMS_sl${seq_len}_pl${pred_len}_dm512_dr0.5_rt${rnn_type}_dw${dec_way}_sl${seg_len}_mae_test_0/"
fi