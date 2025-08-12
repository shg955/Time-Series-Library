export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

# date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id PatchTST_ETTh1_96_96_MS_sample \
  --model PatchTST \
  --data ETTh1 \
  --features MS \
  --target OT \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1 \
  --train_epochs 20 \
  --summarize_only

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id PatchTST_ETTh1_96_192_M_HUFL_exceptOT \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --target HUFL \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 \
  --train_epochs 20 \
  --summarize_only

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id PatchTST_ETTh1_96_336_M_HUFL_exceptOT \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --target HUFL \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1 \
  --train_epochs 20 \
  --summarize_only

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id PatchTST_ETTh1_96_720_M_HUFL_exceptOT \
  --model PatchTST \
  --data ETTh1 \
  --features M \
  --target HUFL \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 16 \
  --itr 1 \
  --train_epochs 20 \
  --summarize_only

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id sample \
  --model PatchTST \
  --data ETTh1 \
  --features MS \
  --target OT \
  --seq_len 90 \
  --label_len 0 \
  --pred_len 20 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 16 \
  --itr 1 \
  --train_epochs 20 \
  --summarize_only