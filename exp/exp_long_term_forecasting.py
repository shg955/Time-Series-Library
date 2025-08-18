from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_attention_matrix, visual_tsne
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import mlflow
from tqdm import tqdm
import glob
import pandas as pd
# import bentoml
from datetime import datetime
from collections import defaultdict
from utils.metrics import MSE, MAE
from utils.PSLoss import PSLoss
from torch.optim import lr_scheduler 


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        # 각 datapoint 수 출력
        # self.print_date_ranges()

        if self.args.task_name == 'long_term_forecast':
            basic_col = [
                "Open",
                "High",
                "Low",
                "Volume",
                "Vwap",
                "snp_index_vwap",
                "snp_index_open",
                "snp_index_high",
                "snp_index_low",
                "snp_index_close",
                "snp_index_volume",
                "Close"]
            self.selected_columns = basic_col

        elif self.args.task_name == 'paper':
            # 경로 합치기
            file_path = os.path.join(self.args.root_path, self.args.data_path)
            df = pd.read_csv(file_path)
            self.data_columns = df.columns.tolist()[1:]
            if self.args.target in self.data_columns:
                self.column_num = self.data_columns.index(self.args.target)

            if self.args.features == 'S':
                self.selected_columns = [self.args.target]
            else:
                self.selected_columns = self.data_columns
            print(self.selected_columns)

        # 저장 디렉토리 설정
        self.suffix = ""
        if self.args.reconstruction:
            self.suffix = "_recon_ps" if self.args.use_ps_loss else "_recon"
        elif self.args.use_ps_loss:
            self.suffix = "_ps" if self.args.use_ps_loss else ""
        elif self.args.position_embedding_emb or self.args.position_embedding_proj:
            self.suffix = "_posiemb"
            if args.position_embedding_weight:
                self.suffix += "_weight"
        elif self.args.position_encoding_emb or self.args.position_encoding_proj:
            self.suffix = "_posienc"
        elif self.args.channelwise_embedding or self.args.channelwise_projection:
            self.suffix = "_channelwise"

        # 경로 설정
        self.checkpoint_dir = os.path.join(
            f"/data/pcw_workspace/Time-Series-Library/checkpoints/{self.args.model}/{self.args.dataset}{self.suffix}/{self.args.model_id}/"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.args.use_ps_loss:
            self.ps_loss_fn = PSLoss(self.model, self.args.patch_len_threshold)
        
        if self.args.is_tsne_emb:
            if self.args.position_encoding_emb or self.args.position_encoding_proj:
                self.embedding_keys = [
                    "before_position_encoding_emb",
                    "after_position_encoding_emb",
                    "before_position_encoding_proj",
                    "after_position_encoding_proj",
                ]
            else:
                self.embedding_keys = [
                    "before_position_embedding_emb",
                    "after_position_embedding_emb",
                    "before_position_embedding_proj",
                    "after_position_embedding_proj",
                ]

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return criterion, mae_criterion

    def print_date_ranges(self):
        print("Checking date ranges for each split:")
        for flag in ['train', 'val', 'test']:
            dataset, _ = self._get_data(flag)
            if hasattr(dataset, 'date_array'):
                dates = dataset.date_array
                print(f"[{flag.upper()}] Date Range: {dates[0]} ~ {dates[-1]} (Total {len(dates)})")
    
    def summarize(self):
        print(f"summarize_only is True: {self.args.model_id}...")

        for flag in ["train", "val", "test"]:
            stats = self.get_feature_losses(flag)
            self.save_feature_loss_to_csv(stats, flag)

    def get_feature_losses(self, flag):
        checkpoint_paths = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint*.pth"))
        checkpoint_list = list(sorted(checkpoint_paths, key=lambda x: int(x.split('_')[-2])))

        if checkpoint_list:
            best_ckpt_path = checkpoint_list[-1]
            print(f"loading model {best_ckpt_path}")
            checkpoint = torch.load(best_ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"No checkpoint found for model_id={self.args.model_id}. Cannot summarize losses.")

        self.model.eval()
        _, loader = self._get_data(flag=flag)
        feature_losses = [[] for _ in range(len(self.selected_columns))]
        mae_losses = [[] for _ in range(len(self.selected_columns))]
        criterion, mae_criterion = self._select_criterion()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="Computing")):

                if len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _, _ = batch
                elif len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

            
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                preds = outputs[:, -self.args.pred_len:, :].detach().cpu()
                trues = batch_y[:, -self.args.pred_len:, :].detach().cpu()

                for idx in range(preds.shape[2]):
                    loss = criterion(preds[:, :, idx], trues[:, :, idx])
                    mae = mae_criterion(preds[:, :, idx], trues[:, :, idx])
                    feature_losses[idx].append(loss.item())
                    mae_losses[idx].append(mae.item())

        stats = []
        for idx in range(len(feature_losses)):
            losses = feature_losses[idx]
            mae_loss = mae_losses[idx]
            mean_loss = round(np.mean(losses), 3)
            std_loss = round(np.std(losses), 3)
            mae_mean_loss = round(np.mean(mae_loss), 3)
            stats.append((self.selected_columns[idx], mae_mean_loss, mean_loss, std_loss))

        return stats
    
    def save_feature_loss_to_csv(self, stats, flag):
        rows = []
        for feature, mae, mean_loss, std_loss in stats:
            row = {
                "model_id": self.args.model_id,
                "feature": feature,
                "mse": mean_loss,
                "mae": mae,
                "std": std_loss,
                "features_type": self.args.features,
                "seq_len": self.args.seq_len,
                "pred_len": self.args.pred_len,
                "flag": flag,
                "model": self.args.model
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        data_name = f"{self.args.dataset}{self.suffix}"
        output_dir = os.path.join(
            "/data/pcw_workspace/Time-Series-Library/loss_results",
            self.args.model,
            data_name
        )
        os.makedirs(output_dir, exist_ok=True)

        # flag 별로 파일 나누기
        csv_path = os.path.join(output_dir, f"{flag}_{self.args.seq_len}_summary.csv")

        # append 또는 새로 생성
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', index=False, header=False)
        else:
            df.to_csv(csv_path, index=False)

        print(f"Saved feature loss to: {csv_path}")

    def vali(self, vali_data, vali_loader, criterion, epoch, flag):
        total_loss = []
        feature_loss = [0.0] * len(self.selected_columns)
        reconstruction_loss = [0.0] * len(self.selected_columns)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(vali_loader, desc="Validation")):

                last_batch = (i == len(vali_loader) - 1)
                flag = 'val' if last_batch else None

                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name = batch
                elif len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    stock_name = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # valid에서는 예측값으로만 loss 계산 (reconstruction도 동일)
                # (M은 모든 feature, MS&S는 하나의 feature)
                if self.args.features == 'M':
                    if self.args.pred_len != 0:
                        valid_pred = outputs[:, -self.args.pred_len:, :] # pred_len
                        valid_true = batch_y[:, -self.args.pred_len:, :].to(self.device) # pred_len
                    else:
                        # pred_len == 0
                        valid_pred = outputs[:, :, :] # seq_len
                        valid_true = batch_x[:, :self.args.seq_len, :].to(self.device) # seq_len
                else:
                    f_dim = self.column_num if self.args.features == 'MS' else 0
                    if self.args.pred_len != 0:
                        valid_pred = outputs[:, -self.args.pred_len:, f_dim:f_dim+1]
                        valid_true = batch_y[:, -self.args.pred_len:, f_dim:f_dim+1].to(self.device)
                    else:
                        # pred_len == 0
                        valid_pred = outputs[:, :, f_dim::f_dim+1] # seq_len
                        valid_true = batch_x[:, :self.args.seq_len, f_dim:f_dim+1].to(self.device) # seq_len
                ## print('valid_pred',valid_pred.shape)
                ## print('valid_true',valid_true.shape)

                # 예측값으로만 loss 계산
                loss = criterion(valid_pred, valid_true)
                total_loss.append(loss.item())

                # feature별 기록용 loss
                # recon일때는 loss 두개 기록해야함 (입력값+예측값 / 예측값)
                if self.args.reconstruction:
                    full_pred = outputs[:, :, :]
                    part1 = batch_x[:, :self.args.seq_len, :] # seq_len
                    if self.args.pred_len > 0:
                        part2 = batch_y[:, -self.args.pred_len:, :] # pred_len
                        full_true = torch.cat([part1, part2], dim=1).to(self.device) # seq_len + pred_len
                    else:
                        full_true = part1.to(self.device)
                else:
                    # recon 아닐때 모든 feature별 기록용 loss
                    full_pred = outputs[:, -self.args.pred_len:, :].detach().cpu()
                    full_true = batch_y[:, -self.args.pred_len:, :].detach().cpu()

                # feature
                for idx in range(full_pred.shape[2]):
                    # recon일때 입력값 + 예측값 loss 계산
                    if self.args.reconstruction:
                        recon_loss = criterion(full_pred[:, :, idx], full_true[:, :, idx])
                        reconstruction_loss[idx] += recon_loss.item()
                    # 예측값 loss 계산
                    ft_loss = criterion(full_pred[:, -self.args.pred_len:, idx],
                                        full_true[:, -self.args.pred_len:, idx])
                    feature_loss[idx] += ft_loss.item()

        # epoch당 feature별 평균 loss
        for idx in range(len(feature_loss)):
            if self.args.features in ['S']:
                prefix = f"feature{self.column_num}"
            else:
                prefix = f"feature{idx}"

            if self.args.reconstruction:
                avg_recon_loss = reconstruction_loss[idx] / len(vali_loader)
                # recon일때 입력값+예측값 기록
                self.writer.add_scalar(
                    f"{prefix} - {self.selected_columns[idx]}/{flag}_loss (input+pred)",
                    avg_recon_loss,
                    epoch + 1,
                )
            # 예측값 기록
            avg_ft_loss = feature_loss[idx] / len(vali_loader)
            self.writer.add_scalar(
                f"{prefix} - {self.selected_columns[idx]}/{flag}_loss",
                avg_ft_loss,
                epoch + 1,
            )

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        print(f"Model Structure:\n{self.model}")
        print(f"Total Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion, mae_criterion = self._select_criterion()
        
        start_epoch = 0
        # 저장된 체크포인트가 있는지 확인

        checkpoint_paths = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint*.pth"))
        checkpoint_list = list(sorted(checkpoint_paths, key=lambda x: int(x.split('_')[-2])))

        if checkpoint_list:
            best_ckpt_path = checkpoint_list[-1]
            print(f"loading model {best_ckpt_path}")
            checkpoint = torch.load(best_ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Checkpoint loaded {best_ckpt_path}. Resuming from epoch {start_epoch}")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        # 체크포인트가 있으면 start부터, 없다면 0부터 학습 시작함
        for epoch in range(start_epoch, self.args.train_epochs):
            self.model.current_epoch = epoch

            iter_count = 0
            base_losses = [] # ps 없는 base loss
            train_loss = [] # 실제 학습 loss
            pred_losses = []
            recon_losses = [] # recon시 seq_len

            if self.args.use_ps_loss:
                ps_stat = {
                    'ps_loss': [],
                    'corr_loss': [],
                    'var_loss': [],
                    'mean_loss': [],
                    'alpha': [],
                    'beta': [],
                    'gamma': [],
                }

            feature_loss = [0.0] * len(self.selected_columns)
            full_feature_loss = [0.0] * len(self.selected_columns)
            recon_feature_loss = [0.0] * len(self.selected_columns)

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(tqdm(train_loader, desc="Training")):

                last_batch = (i == len(train_loader) - 1)
                flag = 'train' if last_batch else None

                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name = batch
                elif len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    stock_name = None
            
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device) # (batch size, seq_len, feature)
                batch_y = batch_y.float().to(self.device) # (batch size, label + pred, feature)
                batch_x_mark = batch_x_mark.float().to(self.device) # (batch size, seq_len, date정보)
                batch_y_mark = batch_y_mark.float().to(self.device) # (batch size, label + pred, date정보)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, _= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # outputs은 recon일때는 seq+pred, 아닐때는 pred만큼만
                # 모델 outputs 원래도 pred_len이였는데 밑에서 한번 더 해주는거임        
        
                # 학습용 feature loss 계산
                # (M은 모든 feature, MS&S는 하나의 feature)
                # 기본 예측값, recon일때는 입력값 + 예측값
                if self.args.reconstruction:
                    if self.args.features == 'M':
                        train_pred = outputs[:, :, :] # seq_len + pred_len
                        part1 = batch_x[:, :self.args.seq_len, :] # seq_len
                        if self.args.pred_len > 0:
                            part2 = batch_y[:, -self.args.pred_len:, :] # pred_len
                            train_true = torch.cat([part1, part2], dim=1).to(self.device) # seq_len + pred_len
                        else:
                            train_true = part1.to(self.device)
                    else:
                        f_dim = self.column_num if self.args.features == 'MS' else 0
                        train_pred = outputs[:, :, f_dim:f_dim+1]
                        part1 = batch_x[:, :self.args.seq_len, f_dim:f_dim+1]
                        if self.args.pred_len > 0:
                            part2 = batch_y[:, -self.args.pred_len:, f_dim:f_dim+1]
                            train_true = torch.cat([part1, part2], dim=1).to(self.device)
                        else:
                            train_true = part1.to(self.device)
                else:
                    if self.args.features == 'M':
                        train_pred = outputs[:, -self.args.pred_len:, :] # pred_len
                        train_true = batch_y[:, -self.args.pred_len:, :].to(self.device) # pred_len
                    else:
                        f_dim = self.column_num if self.args.features == 'MS' else 0
                        train_pred = outputs[:, -self.args.pred_len:, f_dim:f_dim+1]
                        train_true = batch_y[:, -self.args.pred_len:, f_dim:f_dim+1].to(self.device)
                ## print('train_pred', train_pred.shape)
                ## print('train_true', train_true.shape)

                # recon일때는 입력+예측 / 아닐때는 예측
                base_loss = criterion(train_pred, train_true)

                # Add PS Loss
                if self.args.use_ps_loss:
                    psloss, alpha, corr_loss, beta, var_loss, gamma, mean_loss = self.ps_loss_fn.compute_loss(train_true, train_pred)
                    total_loss = base_loss + (psloss * self.args.ps_lambda)

                    ps_stat['ps_loss'].append(psloss.item())
                    ps_stat['corr_loss'].append(corr_loss.item())
                    ps_stat['var_loss'].append(var_loss.item())
                    ps_stat['mean_loss'].append(mean_loss.item())
                    ps_stat['alpha'].append(alpha.item())
                    ps_stat['beta'].append(beta.item())
                    ps_stat['gamma'].append(gamma.item())
                else:
                    total_loss = base_loss

                if self.args.use_ps_loss:
                    base_losses.append(base_loss.item()) # base loss
                train_loss.append(total_loss.item()) # base / ps loss

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                # feature별 기록용 loss (예측값)
                # recon일때는 loss 3개 기록 (입력값+예측값 / 예측값 / 입력값)
                with torch.no_grad():
                    if self.args.pred_len != 0:
                        pred_pred = outputs[:, -self.args.pred_len:, :]
                        pred_true = batch_y[:, -self.args.pred_len:, :].to(self.device)
                        # recon이면 앞에 seq_len 부분만
                        if self.args.reconstruction:
                            recon_pred = outputs[:, :self.args.seq_len, :] # seq_len
                            recon_true = batch_x[:, :self.args.seq_len, :] # seq_len
                    else:
                        # pred_len이 0일 때
                        pred_pred = outputs[:, :self.args.seq_len, :] # seq_len
                        pred_true = batch_x[:, :self.args.seq_len, :] # seq_len

                    # loss 계산할때만 MS target feature 컬럼으로 하면 됨
                    if self.args.features == 'M':
                        pred_loss = criterion(pred_pred, pred_true)
                        pred_losses.append(pred_loss.item())
                        if self.args.reconstruction:
                            recon_loss = criterion(recon_pred, recon_true)
                            recon_losses.append(recon_loss.item())
                    else:
                        f_dim = self.column_num if self.args.features == 'MS' else 0
                        pred_loss = criterion(pred_pred[:, :, f_dim:f_dim+1],
                                              pred_true[:, :, f_dim:f_dim+1])
                        pred_losses.append(pred_loss.item())
                        if self.args.reconstruction:
                            recon_loss = criterion(recon_pred[:, :, f_dim:f_dim+1],
                                                   recon_true[:, :, f_dim:f_dim+1])
                            recon_losses.append(recon_loss.item())

                    # feature별 step loss 기록
                    ###### recon일때 seq_len loss 계산
                    for idx in range(pred_pred.shape[2]):
                        pred_ft_loss = criterion(pred_pred[:, :, idx], pred_true[:, :, idx]) # pred_len
                        feature_loss[idx] += pred_ft_loss.item()
                        
                        if self.args.reconstruction:
                            full_pred = outputs[:, :, idx] # seq_len + pred_len
                            recon_ft_pred = outputs[:, :self.args.seq_len, idx] # seq_len
                            part1 = batch_x[:, :self.args.seq_len, idx] # seq_len
                            if self.args.pred_len > 0:
                                part2 = batch_y[:, -self.args.pred_len:, idx] # pred_len
                                full_true = torch.cat([part1, part2], dim=1).to(self.device) # seq_len + pred_len
                            else:
                                full_true = part1.to(self.device)

                            full_ft_loss = criterion(full_pred, full_true)
                            recon_ft_loss = criterion(recon_ft_pred, part1)

                            full_feature_loss[idx] += full_ft_loss.item()
                            recon_feature_loss[idx] += recon_ft_loss.item()

                        if (i + 1) % 100 == 0:
                            if self.args.features in ['S']:
                                prefix = f"feature{self.column_num}"
                            else:
                                prefix = f"feature{idx}"
                            # 예측 feature step loss
                            self.writer.add_scalar(
                                f"{prefix} - {self.selected_columns[idx]}/train_step_loss",
                                pred_ft_loss.item(),
                                epoch * len(train_loader) + i
                            )
                            # 입력 + 예측값 feature step_loss
                            if self.args.reconstruction:
                                self.writer.add_scalar(
                                    f"{prefix} - {self.selected_columns[idx]}/train_step_loss (input+pred)",
                                    full_ft_loss.item(),
                                    epoch * len(train_loader) + i,
                                )
                                self.writer.add_scalar(
                                    f"{prefix} - {self.selected_columns[idx]}/train_step_loss (recon)",
                                    recon_ft_loss.item(),
                                    epoch * len(train_loader) + i,
                                )

                # 모든 step에서 train step loss 기록
                if (i + 1) % 100 == 0:
                    if self.args.reconstruction:
                        self.writer.add_scalar(
                            "train/step_loss (input+pred)",
                            base_loss.item(),
                            epoch * len(train_loader) + i,
                        )
                        # 예측 step loss
                        self.writer.add_scalar(
                            "train/step_loss",
                            pred_loss.item(),
                            epoch * len(train_loader) + i,
                        )
                        # recon(seq_len) step loss
                        self.writer.add_scalar(
                            "train/step_loss (recon)",
                            recon_loss.item(),
                            epoch * len(train_loader) + i,
                        )
                        if self.args.use_ps_loss:
                            self.writer.add_scalar(
                            "train/step_loss_ps (input+pred)",
                            total_loss.item(),
                            epoch * len(train_loader) + i,
                        )
                    else:
                        # 예측값
                        self.writer.add_scalar(
                            "train/step_loss",
                            base_loss.item(),
                            epoch * len(train_loader) + i,
                        )
                        if self.args.use_ps_loss:
                            self.writer.add_scalar(
                                "train/step_loss_ps",
                                total_loss.item(),
                                epoch * len(train_loader) + i,
                            )

                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {total_loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()
            
            # epoch당 feature별 평균 loss
            for idx in range(len(feature_loss)):
                if self.args.features in ['S']:
                    prefix = f"feature{self.column_num}"
                else:
                    prefix = f"feature{idx}"
                avg_ft_loss = feature_loss[idx] / len(train_loader)
                # 예측값
                self.writer.add_scalar(
                    f"{prefix} - {self.selected_columns[idx]}/train_loss",
                    avg_ft_loss,
                    epoch + 1,
                )
                # 입력 + 예측값
                if self.args.reconstruction:
                    avg_full_ft_loss = full_feature_loss[idx] / len(train_loader) # seq_len + pred_len
                    avg_recon_ft_loss = recon_feature_loss[idx] / len(train_loader) # seq_len
                    self.writer.add_scalar(
                        f"{prefix} - {self.selected_columns[idx]}/train_loss (input+pred)",
                        avg_full_ft_loss,
                        epoch + 1,
                    )
                    self.writer.add_scalar(
                        f"{prefix} - {self.selected_columns[idx]}/train_loss (recon)",
                        avg_recon_ft_loss,
                        epoch + 1,
                    )


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_ps_loss:
                base_loss = np.average(base_losses) # ps x
            train_loss = np.average(train_loss) # ps O

            if self.args.reconstruction:
                pred_loss = np.average(pred_losses)
                recon_loss = np.average(recon_losses)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch, flag="val")
            mlflow.log_metric("vali_loss", vali_loss, step=epoch)
            
            # test_loss = self.vali(test_data, test_loader, criterion)
            # mlflow.log_metric("test_loss", test_loss, step=epoch)

            # 실제 학습용
            # recon일때는 입력+예측 / 아닐때는 예측
            if self.args.reconstruction:
                self.writer.add_scalar("train/loss", pred_loss, epoch+1) # pred_len
                self.writer.add_scalar("train/loss (recon)", recon_loss, epoch+1) # seq_len
                if self.args.use_ps_loss:
                    self.writer.add_scalar("train/loss_ps (input+pred)", train_loss, epoch + 1)
                    self.writer.add_scalar("train/loss (input+pred)", base_loss, epoch + 1)
                else:
                    self.writer.add_scalar("train/loss (input+pred)", train_loss, epoch + 1)
            else:
                if self.args.use_ps_loss:
                    self.writer.add_scalar("train/loss_ps", train_loss, epoch + 1)
                    self.writer.add_scalar("train/loss", base_loss, epoch + 1)
                else:
                    self.writer.add_scalar("train/loss", train_loss, epoch + 1)

            self.writer.add_scalar("val/loss", vali_loss, epoch + 1) # 예측값
            
            if self.args.use_ps_loss:
                for key, values in ps_stat.items():
                    self.writer.add_scalar(f"psloss/{key}", np.average(values), epoch + 1)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, model_optim, self.checkpoint_dir, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        checkpoint_paths = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint*.pth"))
        checkpoint_list = list(sorted(checkpoint_paths, key=lambda x: int(x.split('_')[-2])))
        best_ckpt_path = checkpoint_list[-1]
        print(f"loading model {best_ckpt_path}")

        checkpoint = torch.load(best_ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        input = []
        preds = []
        trues = []
        x_enc_shape, x_mark_enc_shape, x_dec_shape, x_mark_dec_shape = None, None, None, None

        png_folder_path = f"./test_results/{self.args.model}/{self.args.dataset}{self.suffix}/{self.args.model_id}/"
        os.makedirs(png_folder_path, exist_ok=True)

        self.model.eval()
        mse_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):

                if len(batch) == 7:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name, seq_x_dates, seq_y_dates = batch
                elif len(batch) == 6:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, seq_x_dates, seq_y_dates = batch
                    stock_name = self.args.data_path
                elif len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    stock_name = self.args.data_path
                    seq_x_dates, seq_y_dates = None, None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if x_enc_shape is None:
                    x_enc_shape = batch_x.shape
                    x_mark_enc_shape = batch_x_mark.shape # dummy
                    x_dec_shape = dec_inp.shape
                    x_mark_dec_shape = batch_y_mark.shape # dummy
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs_raw = outputs.detach().cpu().numpy()

                # len(attns)
                # attns[0].shape torch.Size([1, 8, 10, 10])
                if self.args.model == 'iTransformer':
                    attn_matrix = attns[-1][0].mean(axis=0).detach().cpu().numpy() # (10, 10)
                

                if self.args.features == 'M':
                    if self.args.pred_len != 0:
                        outputs = outputs[:, -self.args.pred_len:, :] # pred_len
                        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    else:
                        # pred_len == 0
                        outputs = outputs[:, :, :] # seq_len
                        batch_y = batch_x[:, :self.args.seq_len, :].to(self.device)
                else:
                    f_dim = self.column_num if self.args.features == 'MS' else 0
                    if self.args.pred_len != 0:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:f_dim+1]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:f_dim+1].to(self.device)
                    else:
                        outputs = outputs[:, :, f_dim:f_dim+1]
                        batch_y = batch_x[:, -self.args.seq_len:, f_dim:f_dim+1].to(self.device)
                    
                ## print(outputs.shape)
                ## print(batch_y.shape)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 100 == 0:
                    date_labels = []
                    if seq_x_dates is not None and seq_y_dates is not None:
                        seq_x_dates = np.array(seq_x_dates)
                        seq_y_dates = np.array(seq_y_dates)
                               
                        if seq_x_dates.ndim != seq_y_dates.ndim and (self.args.pred_len == 0) :
                            seq_y_dates = np.expand_dims(seq_y_dates, axis=-1)
                        full_dates = np.concatenate([seq_x_dates, seq_y_dates], axis=0)
                        date_labels = [dt[0] for dt in full_dates]
                        date_str = seq_x_dates[0][0] # 'YYYY-MM-DD'
                    else:
                        # batch_x_mark를 날짜 문자열로 변환
                        first_mark = batch_x_mark[0, 0, :3].detach().cpu().numpy()  # [year, month, day]
                        date_str = datetime(int(first_mark[0]), int(first_mark[1]), int(first_mark[2])).strftime("%Y-%m-%d")

                        # seq + pred 길이만큼만
                        for t in range(self.args.seq_len + self.args.pred_len):
                            # batch_x_mark에서 날짜 가져옴
                            if t < batch_x_mark.shape[1]:
                                mark = batch_x_mark[0, t, :3]
                            # batch_y_mark에서 날짜 가져오면서 인덱스 조정
                            else:
                                mark = batch_y_mark[0, t - batch_x_mark.shape[1], :3]
                            # 년, 월, 일 정보
                            y, m, d = mark.detach().cpu().numpy()
                            date = datetime(int(y), int(m), int(d))
                            date_labels.append(date.strftime('%y-%m-%d'))

                    # Attention matrix 시각화
                    if not self.args.dataset in ["traffic", "ECL"]:
                        if self.args.model == 'iTransformer':
                            visual_attention_matrix(
                                attn_matrix = attn_matrix,
                                save_root ='/data/pcw_workspace/Time-Series-Library/test_results_matrix',
                                suffix = self.suffix,
                                model = self.args.model,
                                dataset_name = self.args.dataset,
                                model_id = self.args.model_id,
                                target_feature = self.args.target,
                                feature_names = self.data_columns,
                                freq = self.args.freq,
                                features = self.args.features,
                                sample_idx=i
                            )

                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                    if self.args.features == 'M':
                        input = input[0, :, :]
                    if self.args.features == 'MS':
                        f_dim = self.column_num if self.args.features == 'MS' else 0
                        input = input[0, :, f_dim:f_dim+1]
                    
                    ## print(input.shape)
                    ## print(true.shape)
                    ## print(pred.shape)
                    
                    input_2d = input[0] if input.ndim == 3 else input

                    # pred_len != 0일때
                    if self.args.reconstruction:
                        if self.args.pred_len == 0:
                            gt = input_2d.copy()
                            gt = gt[:, np.newaxis]
                        else:
                            gt = np.concatenate((input_2d, true[0, :, :]), axis=0)

                        if self.args.features == 'MS':
                            f_dim = self.column_num
                            pd = outputs_raw[0, :, f_dim:f_dim+1]
                        else:
                            pd = outputs_raw[0, :, :]
                    else:
                        gt = np.concatenate((input_2d, true[0, :, :]), axis=0)
                        pd = np.concatenate((input_2d, pred[0, :, :]), axis=0)
                        
                    
                    num_features = gt.shape[1]
                    # 파일 이름 지정
                    if isinstance(stock_name, (list, tuple, np.ndarray)):
                        stock_str = str(stock_name[0])
                    else:
                        stock_str = os.path.splitext(str(stock_name))[0]

                    # traffic, ECL은 OT만 시각화
                    if self.args.dataset in ["traffic", "ECL"]:
                        # target feature 인덱스만 시각화
                        if "OT" in self.selected_columns:
                            f_idx_list = [self.selected_columns.index("OT")]
                        else:
                            print("Warning: OT not in selected_columns.")
                            f_idx_list = []
                    else:
                        f_idx_list = list(range(num_features))

                    for f_idx in f_idx_list:
                        if self.args.features != 'M':
                            feature_name = self.args.target
                        else:
                            feature_name = self.selected_columns[f_idx]
                        # feature별 하위 디렉토리 생성
                        feature_folder = os.path.join(png_folder_path, feature_name)
                        os.makedirs(feature_folder, exist_ok=True)

                        file_name = f"{stock_str}_{date_str}_{feature_name}_{i}.png"
                        
                        mse=visual(
                            true=gt[:,f_idx],
                            pred=pd[:, f_idx],
                            name = os.path.join(feature_folder, file_name),
                            stock_str=f"{stock_str} - {feature_name}",
                            date_labels=date_labels,
                            pred_len = self.args.pred_len,
                            is_reconstruction=self.args.reconstruction)
                        
                        # mse list 저장
                        mse_list.append((feature_name, mse, os.path.join(feature_name, file_name)))
                    # MLflow에 artifact로 저장
                    #mlflow.log_artifact(os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_folder_path = (f"./results/{self.args.model}/{self.args.dataset}{self.suffix}/{self.args.model_id}/")
        os.makedirs(result_folder_path, exist_ok=True)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        total_mae, total_mse, rmse, mape, mspe = metric(preds, trues)
        print('Total mse:{}, mae:{}, dtw:{}'.format(total_mse, total_mae, dtw))
        np.save(result_folder_path + 'metrics.npy', np.array([total_mae, total_mse, rmse, mape, mspe]))
        np.save(result_folder_path + 'pred.npy', preds)
        np.save(result_folder_path + 'true.npy', trues)

        # feature별 metric 리스트 만들기
        feature_metrics = []
        num_features = preds.shape[-1]

        for f_idx in range(num_features):
            preds_f = preds[:, :, f_idx]
            trues_f = trues[:, :, f_idx]
            mae_f, mse_f, _, _, _= metric(preds_f, trues_f)
            feature_metrics.append((mse_f, mae_f))

        #######
        feature_mse_dict = defaultdict(list)
        for feature_name, mse, file_path in mse_list:
            feature_mse_dict[feature_name].append((mse, file_path))

        mse_folder_path = os.path.join("mse_results", self.args.model, f"{self.args.dataset}{self.suffix}", self.args.model_id)
        os.makedirs(mse_folder_path, exist_ok=True)

        # feature별로 저장
        for f_idx, (feature, entries) in enumerate(feature_mse_dict.items()):
            entries.sort()
            mse_f, mae_f = feature_metrics[f_idx]

            file_path = os.path.join(mse_folder_path, f"{feature}_mse_result.txt")

            with open(file_path, "a") as f:
                f.write(f"{setting}\n")
                f.write(f"Total mse:{total_mse}, mae:{total_mae}\n\n")
                f.write(f"Feature mse:{mse_f:.6f}, mae:{mae_f:.6f}\n\n")

                f.write("Best 3 MSE (Lowest MSE):\n")
                for mse, file_path_str in entries[:3]:
                    f.write(f"file_name = {file_path_str}, MSE = {mse:.6f}\n")

                f.write("\nWorst 3 MSE (Highest MSE):\n")
                for mse, file_path_str in entries[-3:]:
                    f.write(f"file_name = {file_path_str}, MSE = {mse:.6f}\n")

                f.write("\nMSE list(sort):\n")
                for mse, file_path_str in entries:
                    f.write(f"file_name = {file_path_str}, MSE = {mse:.6f}\n")

        return
    
    ############# tsne 관련 코드
    def init_embedding(self):
        return {key: [] for key in self.embedding_keys}
    
    def store_embeddings(self, embeddings, embedding_dict):
        for idx, embed in enumerate(embeddings):
            if embed is not None:
                embedding_dict[self.embedding_keys[idx]].append(embed.detach().cpu())

    def tsne_emb(self, setting):
        checkpoint_paths = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint*.pth"))
        checkpoint_list = list(sorted(checkpoint_paths, key=lambda x: int(x.split('_')[-2])))

        save_path = os.path.join(
            "/data/pcw_workspace/Time-Series-Library/tsne_output",
            self.args.model,
            f"{self.args.dataset}{self.suffix}",
            self.args.model_id
        )
        os.makedirs(save_path, exist_ok=True)
        
        ## val은 해당되는 에폭 모두, test는 마지막 에폭만
        for flag in ["val", "test"]:
            print(f"[{flag}] save tsne.png is True: {self.args.model_id}...")

            _, loader = self._get_data(flag=flag)

            # val은 전체, test는 마지막 하나만
            loop_ckpts = checkpoint_list if flag == "val" else [checkpoint_list[-1]]

            for ckpt_path in loop_ckpts:
                current_epoch = ckpt_path.split('_')[-2]
                print(f"Loading model from epoch {current_epoch} → {ckpt_path}")

                checkpoint = torch.load(ckpt_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])

                embedding_dict = self.init_embedding()
                self.model.eval()
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(loader, desc=f"Inference ({flag}, epoch {current_epoch})")):
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]

                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)

                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                        # 모델 추론
                        embeddings = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        self.store_embeddings(embeddings, embedding_dict)

                # positional embedding 전후 비교 값
                for key in tqdm(self.embedding_keys, desc=f"Embedding Visualization"):
                    if not embedding_dict[key]:
                        continue

                    # Batch 축으로 붙임
                    embed_tensor = torch.cat(embedding_dict[key], dim=0)  # [N_total, F, D]

                    if embed_tensor.dim() != 3:
                        raise ValueError(f"Expected embedding tensor to have 3 dimensions [N, F, D], but got {embed_tensor.shape}")
                    
                    N, F, D = embed_tensor.shape
                
                    labels = torch.linspace(0, F-1, F).to(torch.int).view(1, F).expand(N, F).cpu().flatten()

                    embed_tensor = embed_tensor.contiguous().view(-1, embed_tensor.size(-1))

                    embed_np = embed_tensor.cpu().numpy()  # [F, D]
                    
                    # embedding 값 시각화
                    visual_tsne(
                        vectors=embed_np,
                        flag=flag,
                        save_path=save_path,
                        filename_prefix=key,
                        labels = labels,
                        epoch=current_epoch,
                        model=self.args.model,
                        dataset=self.args.dataset,
                        model_id=self.args.model_id
                    )

                # positional embedding vector 값
                if not (self.args.position_encoding_emb or self.args.position_encoding_proj):
                    for name in tqdm(["position_embedding_em", "position_embedding_pr", "position_embedding_sh"], desc=f"Vector Visualization"):
                        tensor = getattr(self.model, name, None)
                        if tensor is not None:
                            # F = feature+date 수
                            labels = torch.arange(tensor.shape[0])  # shape: [F]

                            visual_tsne(
                                vectors=tensor.detach().cpu(),
                                flag=flag,
                                save_path=save_path,
                                filename_prefix=name,
                                labels = labels,
                                epoch=current_epoch,
                                model=self.args.model,
                                dataset=self.args.dataset,
                                model_id=self.args.model_id
                            )

                # L1 통계 저장
                self.save_embedding_to_csv(
                    embedding_dict=embedding_dict,
                    epoch=current_epoch,
                    flag=flag
                )

    def save_embedding_to_csv(self, embedding_dict, epoch, flag):
        output_dir = os.path.join(
            "/data/pcw_workspace/Time-Series-Library/embedding_vector",
            self.args.model,
            f"{self.args.dataset}{self.suffix}"
        )
        os.makedirs(output_dir, exist_ok=True)

        save_items = []  # [(rows, filename)]

        model_rows = []
        pos_rows = []

        # ----- self.embedding_keys: L1 통계 -----
        for key in tqdm(self.embedding_keys, desc="Embedding Vector L1 Stats"):
            if not embedding_dict.get(key):
                continue

            embed_tensor = torch.cat(embedding_dict[key], dim=0)  # [N, F, D]

            l1_values = embed_tensor.abs().sum(dim=2)  # [N, F] ← 각 feature별 L1 norm
            l1_mean = l1_values.mean(dim=0) # [F] ← feature별 평균 L1 norm
            l1_std  = l1_values.std(dim=0)
            l1_max  = l1_values.max(dim=0).values
            l1_min  = l1_values.min(dim=0).values

            for i in range(l1_mean.shape[0]):
                model_rows.append({
                    'model_id' : self.args.model_id,
                    'type': 'model embedding',
                    'name': key,
                    'feature_index': i,
                    'epoch': epoch,
                    'l1_mean': l1_mean[i].item(),
                    'l1_std': l1_std[i].item(),
                    'l1_max': l1_max[i].item(),
                    'l1_min': l1_min[i].item(),
                    'flag': flag,
                    'model' : self.args.model
                })

        save_items.append((model_rows, f"{flag}_model_embedding.csv"))

        # ----- position_embedding_*: 평균, 분산, L1 norm -----
        if not (self.args.position_encoding_emb or self.args.position_encoding_proj):
            for name in ["position_embedding_em", "position_embedding_pr", "position_embedding_sh"]:
                pe_tensor = getattr(self.model, name, None)
                if pe_tensor is not None:
                    pe_tensor = pe_tensor.detach().cpu()  # [F, D]
                    l1_sum = pe_tensor.abs().sum(dim=1)   # [F]

                    for i, l1 in enumerate(l1_sum):
                        pos_rows.append({
                            'model_id' : self.args.model_id,
                            'type': 'position embedding',
                            'name': name,
                            'feature_index': i,
                            'epoch': epoch,
                            'l1_sum': l1.item(),
                            'flag': flag,
                            'model' : self.args.model
                        })

        save_items.append((pos_rows, f"{flag}_position_embedding.csv"))
        
        for rows, filename in save_items:
            if not rows:
                continue

            df = pd.DataFrame(rows)
            csv_path = os.path.join(output_dir, filename)

            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', index=False, header=False)
            else:
                df.to_csv(csv_path, index=False)

            print(f"[{flag}-Epoch{epoch}] Saved to: {csv_path}")