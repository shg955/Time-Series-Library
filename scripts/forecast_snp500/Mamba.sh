# 1 train, 0 test
python -u run.py \
  --is_training 0 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/Input/ \
  --model_id Mamba_sample \
  --model Mamba \
  --data SNP500 \
  --batch_size 1024 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 12 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 12 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1

#   --model_id Mamba_Apr24_96_24_48 \