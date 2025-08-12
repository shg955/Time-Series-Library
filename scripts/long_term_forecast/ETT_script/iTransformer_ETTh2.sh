model_name=iTransformer

# date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT

python -u run.py \
  --task_name paper \
  --is_training 1 \
  --root_path /data/pcw_workspace/Time-Series-Library/dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id iTransformer_ETTh2_96_96_MS_LUFL_recon_ps \
  --model iTransformer \
  --data ETTh2 \
  --features MS \
  --target LUFL \
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
  --train_epochs 20 \
  --reconstruction \
  --use_ps_loss \
  --summarize_only \

          