# Model params below need to be set in WPMixer.py Line 15, instead of this script
wavelets=(sym3 coif5 sym4 db2)
levels=(2 3 1 2)
tfactors=(3 7 5 7)
dfactors=(5 5 7 8)
strides=(8 8 8 8)


# 1 train, 0 test
python -u run.py \
	--is_training 0 \
	--task_name long_term_forecast \
	--root_path /data/pcw_workspace/Time-Series-Library/dataset/DAN_Dataset_11_07/Input/ \
	--model_id WPMixer_Apr29_96_24_coif4_4 \
	--model WPMixer \
	--data SNP500 \
	--seq_len 96 \
	--pred_len 24 \
	--label_len 0 \
	--d_model 32 \
	--patch_len 16 \
	--batch_size 1024 \
	--c_out 12 \
	--use_amp

# 	--model_id WPMixer_Apr29_96_24_coif4_4 \
#   --model_id WPMixer_Apr29_96_24_db3_3 \