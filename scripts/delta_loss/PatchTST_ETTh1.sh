export CUDA_VISIBLE_DEVICES=2

model_name=PatchTST

for pred_len in 96 192 336 720; do
  log_dir=logs/${model_name}/ETTh1
  mkdir -p "$log_dir"
  log_file=${log_dir}/96_${pred_len}.log

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 336 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 3 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --train_epochs 100 \
    --patience 30 \
    --n_heads 4 \
    --d_model 16 \
    --d_ff 128 \
    --batch_size 128 \
    --lradj 'type3' \
    --itr 1 > "$log_file" 2>&1
done

# for pred_len in 96 192 336 720; do
#   log_dir=logs/${model_name}/ETTm1
#   mkdir -p "$log_dir"
#   log_file=${log_dir}/96_${pred_len}.log

#   python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_96_${pred_len} \
#     --model $model_name \
#     --data ETTm1 \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --train_epochs 30 \
#     --patience 10 \
#     --itr 1 > "$log_file" 2>&1
# done

# for pred_len in 96 192 336 720; do
#   log_dir=logs/${model_name}/Exchange
#   mkdir -p "$log_dir"
#   log_file=${log_dir}/96_${pred_len}.log

#   python -u run.py \
#     --task_name long_term_forecast \
#     --is_training 1 \
#     --root_path ./dataset/exchange_rate/ \
#     --data_path exchange_rate.csv \
#     --model_id Exchange_96_${pred_len} \
#     --model $model_name \
#     --data custom \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len $pred_len \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 8 \
#     --dec_in 8 \
#     --c_out 8 \
#     --des 'Exp' \
#     --train_epochs 30 \
#     --patience 10 \
#     --itr 1 > "$log_file" 2>&1
# done