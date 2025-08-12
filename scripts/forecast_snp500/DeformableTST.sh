# 1 train, 0 test
python -u run.py \
    --is_training 0 \
    --task_name long_term_forecast \
    --root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/Input/ \
    --model_id DeformableTST_May07_96_24 \
    --model DeformableTST \
    --data SNP500 \
    --features M \
    --enc_in 12 \
    --seq_len 96 \
    --pred_len 24 \
    --label_len 0 \
    --batch_size 1024 \
    --use_amp