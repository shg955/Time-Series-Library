# 1 train, 0 test
# --embed learned (test할때)
python -u run.py \
  --is_training 0 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/Input/ \
  --model_id FEDformer_Apr07_96_24_48 \
  --model FEDformer \
  --data SNP500 \
  --batch_size 1024 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --label_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Quant_snp/dataset/DAN_Dataset_11_07/sample/ \
  --model_id sample \
  --model FEDformer \
  --data SNP500 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --label_len 48 \
  --train_epochs 2 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --des 'Exp' \
  --itr 1