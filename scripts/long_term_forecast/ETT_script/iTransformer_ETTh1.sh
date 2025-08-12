qorhvkexport CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id iTransformer_ETTh1_96_96_test \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --target OT \
  --seq_len 96 \
  --pred_len 96 \
  --label_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --train_epochs 1 \
  --projection \
  --reconstruction \
  --summarize_only \
  --use_ps_loss \
  > log/iTransformer/M/ETTh1_96_96_M.txt &

python -u run.py \
  --task_name paper \
  --is_training 0 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id iTransformer_ETTh1_96_192_M_inverse \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --inverse \
  --train_epochs 20 \
  --reconstruction \
  --summarize_only \
  --use_ps_loss \
  > log/iTransformer/M/ETTh1_96_192_M.txt &

python -u run.py \
  --task_name paper \
  --is_training 0 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id iTransformer_ETTh1_96_336_M_inverse \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --inverse \
  --train_epochs 20 \
  --reconstruction \
  --summarize_only \
  --use_ps_loss \
  > log/iTransformer/M/ETTh1_96_336_M.txt &

python -u run.py \
  --task_name paper \
  --is_training 0 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id iTransformer_ETTh1_96_720_M_inverse \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --target OT \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --inverse \
  --train_epochs 20 \
  --reconstruction \
  --summarize_only \
  --use_ps_loss \
  > log/iTransformer/M/ETTh1_96_720_M.txt &