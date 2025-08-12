# 1 train, 0 test
python -u run.py \
    --is_training 1 \
    --task_name paper \
    --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
    --data_path exchange_rate.csv \
    --model_id DeformableTST_Exchange_96_96 \
    --model DeformableTST \
    --data custom \
    --features M \
    --enc_in 8 \
    --seq_len 96 \
    --pred_len 96 \
    --label_len 0 \
    --train_epochs 50 \
    --target OT \
    --use_amp

python -u run.py \
    --is_training 1 \
    --task_name paper \
    --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
    --data_path exchange_rate.csv \
    --model_id DeformableTST_Exchange_96_192 \
    --model DeformableTST \
    --data custom \
    --features M \
    --enc_in 8 \
    --seq_len 96 \
    --pred_len 192 \
    --label_len 0 \
    --train_epochs 50 \
    --target OT \
    --use_amp

python -u run.py \
    --is_training 1 \
    --task_name paper \
    --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
    --data_path exchange_rate.csv \
    --model_id DeformableTST_Exchange_96_336 \
    --model DeformableTST \
    --data custom \
    --features M \
    --enc_in 8 \
    --seq_len 96 \
    --pred_len 336 \
    --label_len 0 \
    --train_epochs 50 \
    --target OT \
    --use_amp

python -u run.py \
    --is_training 1 \
    --task_name paper \
    --root_path /data/pcw_workspace/Time-Series-Library/dataset/exchange_rate \
    --data_path exchange_rate.csv \
    --model_id DeformableTST_Exchange_96_720 \
    --model DeformableTST \
    --data custom \
    --features M \
    --enc_in 8 \
    --seq_len 96 \
    --pred_len 720 \
    --label_len 0 \
    --train_epochs 50 \
    --target OT \
    --use_amp

# pred_len 96, 192, 336, 720