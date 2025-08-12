# date,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,OT

import subprocess
import os
from datetime import datetime

base_args = [
    "--task_name", "paper",
    "--root_path", "/data/pcw_workspace/Time-Series-Library/dataset/electricity/",
    "--data_path", "electricity.csv",
    "--model", "iTransformer",
    "--data", "custom",
    "--seq_len", "96",
    "--label_len", "0",
    "--e_layers", "3",
    "--d_layers", "1",
    "--factor", "3",
    "--des", "Exp",
    "--d_model", "512",
    "--d_ff", "512",
    "--itr", "1",
    "--train_epochs", "20",
    "--batch_size", "16",
    "--learning_rate", "0.0005"
]

# 실험 설정
pred_lens = [96, 192, 336, 720]
feature_settings = []

# OT 추가
# for prefix in ["M", "MS", "S"]:
#     feature_settings.append((prefix, "OT"))

# 0 ~ 319 숫자 추가
for prefix in ["MS", "S"]:
    for i in range(320):  # 0 ~ 319 포함
        feature_settings.append((prefix, str(i)))

# 실험 조합: 기본 / recon / recon + ps
extra_option_sets = [
    [],  # 기본
    ["--reconstruction"],  # reconstruction only
    ["--reconstruction", "--use_ps_loss"]  # reconstruction + ps_loss
]

data_path_idx = base_args.index("--data_path") + 1
data_path_value = base_args[data_path_idx]
dataset_name = os.path.splitext(data_path_value)[0]

log_dir = os.path.join("log", "iTransformer", dataset_name)
os.makedirs(log_dir, exist_ok=True)

# 실행 반복
for features, target in feature_settings:
    for pred_len in pred_lens:

        suffix = f"{features}_{target}" if not (features == "M" and target == "OT") else "M"
        base_model_id = f"iTransformer_ECL_96_{pred_len}_{suffix}"

        # features가 S일 때는 1, 아닐 때는 321
        enc_dec_c_out = "1" if features == "S" else "321"

        for extra_opts in extra_option_sets:
            # model_id suffix 설정
            if "--reconstruction" in extra_opts and "--use_ps_loss" in extra_opts:
                opt_suffix = "_recon_ps"
            elif "--reconstruction" in extra_opts:
                opt_suffix = "_recon"
            else:
                opt_suffix = ""

            model_id = base_model_id + opt_suffix

            common_args = [
                "--model_id", model_id,
                "--features", features,
                "--target", target,
                "--pred_len", str(pred_len),
                "--enc_in", enc_dec_c_out,
                "--dec_in", enc_dec_c_out,
                "--c_out", enc_dec_c_out
            ]

            cmd_train = ["python", "-u", "run.py"] + common_args + ["--is_training", "1"] + base_args + extra_opts
            cmd_summarize = ["python", "-u", "run.py"] + common_args + ["--is_training", "0", "--summarize_only"] + base_args + extra_opts

            log_file_name = f"{features}.txt"
            log_path = os.path.join(log_dir, log_file_name)

            with open(log_path, "a") as f:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n\n=== TRAIN: {model_id} @ {now} ===\n")
                subprocess.run(cmd_train, stdout=f, stderr=subprocess.STDOUT)

                f.write(f"\n\n=== SUMMARIZE: {model_id} @ {now} ===\n")
                subprocess.run(cmd_summarize, stdout=f, stderr=subprocess.STDOUT)
