export CUDA_VISIBLE_DEVICES=2

model_name=iTransformer

alpha=0.6
beta=0.2
gamma=0.2

for pred_len in 96 192 336 720; do
  if [[ "$pred_len" -eq 96 || "$pred_len" -eq 192 ]]; then
    d_model=256
    d_ff=256
  else
    d_model=512
    d_ff=512
  fi

  log_dir=logs/${model_name}_use_delta_loss_PACF_${alpha}_${beta}_${gamma}/ETTh1
  mkdir -p "$log_dir"
  log_file=${log_dir}/96_${pred_len}.log

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_${pred_len}_use_delta_loss_PACF_${alpha}_${beta}_${gamma} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --train_epochs 30 \
    --patience 20 \
    --d_model $d_model \
    --d_ff $d_ff \
    --use_delta_loss_PACF 1 \
    --alpha $alpha \
    --beta $beta \
    --gamma $gamma \
    --itr 1 > "$log_file" 2>&1
done
