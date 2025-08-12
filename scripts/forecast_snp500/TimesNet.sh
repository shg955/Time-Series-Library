# 1 train, 0 test
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/Input/ \
  --model_id TimesNet_Apr17_96_24 \
  --model TimesNet \
  --data SNP500 \
  --batch_size 1024 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --label_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1

# label 원래 48
#   --model_id TimesNet_Apr17_96_24 \