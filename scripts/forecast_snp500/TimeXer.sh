model_name=TimeXer
des='Timexer-MS'
patch_len=24
# enc_in, dec_in, c_out date 뺀 feature 수만큼 설정해줘야함

# 1 train, 0 test
python3 -u run.py \
  --is_training 0 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Quant_snp/dataset/DAN_Dataset_11_07/Input/ \
  --model_id TimeXer_Apr01_96_24 \
  --model TimeXer \
  --data SNP500 \
  --batch_size 1024 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --enc_in 12\
  --dec_in 12\
  --c_out 12\
  --des 'Exp' \
  --patch_len 24 \
  --itr 1

# sample
python3 -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Quant_snp/dataset/DAN_Dataset_11_07/sample/ \
  --model_id sample \
  --model TimeXer \
  --data SNP500 \
  --train_epochs 2 \
  --batch_size 256 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --enc_in 12\
  --dec_in 12\
  --c_out 12\
  --des 'Exp' \
  --patch_len 24 \
  --itr 1
