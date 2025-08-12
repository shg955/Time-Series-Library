export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# date,0,1,2,3,4,5,6,OT

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_96_M_test \
  --model iTransformer \
  --data custom \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --train_epochs 1 \
  --reconstruction \
  --use_ps_loss \
  --summarize_only \
  > log/iTransformer/M/Exchange_96_96_M.txt &

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_192_M \
  --model iTransformer \
  --data custom \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --train_epochs 20 \
  --reconstruction \
  --use_ps_loss \
  --summarize_only \
  > log/iTransformer/M/Exchange_96_192_M.txt &

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_336_M \
  --model iTransformer \
  --data custom \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --train_epochs 20 \
  --reconstruction \
  --use_ps_loss \
  --summarize_only \
  > log/iTransformer/M/Exchange_96_336_M.txt &

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id iTransformer_Exchange_96_720_M \
  --model iTransformer \
  --data custom \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --train_epochs 20 \
  --reconstruction \
  --use_ps_loss \
  --summarize_only \
  > log/iTransformer/M/Exchange_96_720_M.txt &
