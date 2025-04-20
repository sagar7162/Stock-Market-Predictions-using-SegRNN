#!/bin/bash
# Universal stock prediction script that works with any stock CSV file
# Usage: sh scripts/SegRNN/stock_predict.sh --stock STOCKNAME [--is_training 0/1] [--skip_test 0/1]

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Default parameter values
stock_name="BPCL"  # Default stock
root_path_name=./dataset/ 
data_name=custom
features=MS
target=Close
freq=d
model_name=SegRNN
rnn_type=gru
dec_way=pmf
seg_len=48
win_len=48
seq_len=720
pred_len=96
is_training=1  # Default is testing mode
skip_test=0    # Default is to run test phase
enc_in=5       # Updated: 5 columns are actually being loaded
dec_in=5       # Updated: keeping consistent with enc_in
c_out=1        # 1 output feature (Close)

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stock) stock_name="$2"; shift 2;;
    --is_training) is_training="$2"; shift 2;;
    --skip_test) skip_test="$2"; shift 2;;
    --pred_len) pred_len="$2"; shift 2;;
    --seq_len) seq_len="$2"; shift 2;;
    --model) model_name="$2"; shift 2;;
    *) echo "Unknown parameter passed: $1"; exit 1;;
  esac
done

# Set data path based on stock name
data_path_name="${stock_name}.csv"
model_id_name="${stock_name}"

# Validate that the CSV file exists
if [ ! -f "${root_path_name}${data_path_name}" ]; then
  echo "Error: Stock CSV file ${root_path_name}${data_path_name} not found!"
  echo "Available stock files:"
  ls -1 ${root_path_name}*.csv
  exit 1
fi

echo "Running prediction for stock: ${stock_name}"
echo "Parameters: is_training=${is_training}, skip_test=${skip_test}, pred_len=${pred_len}, seq_len=${seq_len}"

# Create the model_id string properly (without single quotes)
full_model_id="${model_id_name}_${seq_len}_${pred_len}"

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
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 30 \
  --patience 1 \
  --rnn_type $rnn_type \
  --dec_way $dec_way \
  --dec_in $dec_in \
  --c_out $c_out \
  --channel_id 1 \
  --itr 1 --batch_size 256 --learning_rate 0.001 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

result=$?
if [ $result -ne 0 ]; then
  echo "Error: Script execution failed with error code $result. Check the log for details."
  tail -20 logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
  exit $result
fi

# Display helpful message about results location
results_dir="./results/${full_model_id}_${model_name}_custom_ftMS_sl${seq_len}_pl${pred_len}_dm512_dr0.5_rt${rnn_type}_dw${dec_way}_sl${seg_len}_mae_test_0/"
echo "Processing completed for ${stock_name}."
echo "Results are available in: ${results_dir}"

# Check if results directory exists and show available files
if [ -d "$results_dir" ]; then
  echo "Available result files:"
  ls -1 "$results_dir"
  
fi