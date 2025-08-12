import subprocess
import os
from datetime import datetime

base_args = [
    "--task_name", "paper",
    "--root_path", "/data/pcw_workspace/Time-Series-Library/dataset/ETT-small/",
    "--data_path", "ETTm2.csv",
    "--model", "iTransformer",
    "--data", "ETTm2",
    "--seq_len", "96",
    "--label_len", "0",
    "--e_layers", "2",
    "--d_layers", "1",
    "--factor", "3",
    "--des", "Exp",
    "--d_model", "128",
    "--d_ff", "128",
    "--itr", "1",
    "--train_epochs", "20"
]

# 실험 설정
pred_lens = [96, 192, 336, 720]
feature_settings = [
    ("M", "OT"),
    ("MS", "OT"),
    ("MS", "HUFL"),
    ("MS", "HULL"),
    ("MS", "MUFL"),
    ("MS", "MULL"),
    ("MS", "LUFL"),
    ("MS", "LULL"),
    ("S", "OT"),
    ("S", "HUFL"),
    ("S", "HULL"),
    ("S", "MUFL"),
    ("S", "MULL"),
    ("S", "LUFL"),
    ("S", "LULL")
]

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
        base_model_id = f"iTransformer_ETTm2_96_{pred_len}_{suffix}"

        # features가 S일 때는 1, 아닐 때는 7
        enc_dec_c_out = "1" if features == "S" else "7"

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
