# model_name=iTransformer
# T/F typeX 1,0으로 변환
# 1 train, 0 test
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/Input/ \
  --model_id iTransformer_May20_96_24 \
  --model iTransformer \
  --data SNP500 \
  --batch_size 1024 \
  --seq_len 96 \
  --pred_len 24 \
  --label_len 0 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --itr 1

  # --iaaft --jitter --timeGAN

# test
python -u run.py \
  --is_training 0 \
  --root_path /data/pcw_workspace/Quant_snp/dataset/DAN_Dataset_11_07/Input/ \
  --model_id Mar10_shuf_T_96_24 \
  --model iTransformer \
  --data SNP500 \
  --batch_size 1 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --des 'Exp' \
  --itr 1 \
  --inverse

# STFT
python -u run.py \
  --is_training 0 \
  --root_path /data/pcw_workspace/Quant_snp/dataset/DAN_Dataset_11_07/Input/ \
  --model_id Mar25_stft_96_24_4 \
  --model iTransformer \
  --data SNP500 \
  --batch_size 1024 \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --des 'Exp' \
  --itr 1 \
  --hop_length 1 \
  --n_fft 4


python -u run.py \
  --is_training 0 \
  --task_name long_term_forecast \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/sample/ \
  --model_id sample \
  --model iTransformer \
  --data SNP500 \
  --train_epochs 2 \
  --features M \
  --seq_len 90 \
  --pred_len 60 \
  --e_layers 2 \
  --enc_in 12 \
  --dec_in 12 \
  --c_out 12 \
  --itr 1

# writer
# 96, 24