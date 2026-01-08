from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
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
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# def delta_loss(y_pred: torch.Tensor, y_true: torch.Tensor, k: int) -> torch.Tensor:
#     """
#     y_pred, y_true: [B, T, C] 또는 [B, T]
#     k-step delta matching L1 loss
#     """
#     if y_true.size(1) <= k:
#         # pred_len이 k보다 짧으면 계산 불가 → 0 처리(또는 skip)
#         return y_true.new_tensor(0.0)

#     y_true_d = y_true[:, k:] - y_true[:, :-k]
#     y_pred_d = y_pred[:, k:] - y_pred[:, :-k]

#     return (y_true_d - y_pred_d).abs().mean()

def delta_loss(y_pred: torch.Tensor, y_true: torch.Tensor, k: int) -> torch.Tensor:
    """
    교수님 방식:
    - i < k (즉 i-k < 0) 구간: |y_i - y_hat_i| 로 대체 (L1)
    - i >= k 구간: |(y_i - y_{i-k}) - (y_hat_i - y_hat_{i-k})| (기존 delta L1)
    최종적으로 time축 전체(T)에 대해 평균
    """
    T = y_true.size(1)
    if T == 0:
        return y_true.new_tensor(0.0)

    # 1) i < k 구간: L1(y, yhat)
    head_len = min(k, T)  # T < k일 수도 있으니 안전하게
    head_l1 = (y_true[:, :head_len] - y_pred[:, :head_len]).abs()  # [B, head_len, ...] or [B, head_len]

    # 2) i >= k 구간: delta L1
    if T > k:
        y_true_d = y_true[:, k:] - y_true[:, :-k]
        y_pred_d = y_pred[:, k:] - y_pred[:, :-k]
        tail_l1 = (y_true_d - y_pred_d).abs()  # [B, T-k, ...]
        all_terms = torch.cat([head_l1, tail_l1], dim=1)  # time dim concat -> [B, T, ...]
    else:
        # 전부 i<k라서 delta 항이 없음 -> L1만
        all_terms = head_l1

    return all_terms.mean()


class Exp_Delta_Loss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Delta_Loss, self).__init__(args)
        self.writer = SummaryWriter(log_dir=('./tensorboard_log/'+args.model+'_'+args.model_id))

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
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_d1 = []
        total_dn = []
        total_mse_list = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
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
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                alpha, beta, gamma = self.args.alpha, self.args.beta, self.args.gamma

                mse = criterion(pred, true)
                d1  = delta_loss(pred, true, k=1)
                dn = delta_loss(pred, true, k=self.args.delta_n)

                loss = alpha * mse + beta * d1 + gamma * dn

                total_loss.append(loss.item())
                total_d1.append(d1.item())
                total_dn.append(dn.item())
                total_mse_list.append(mse.item())

        self.model.train()
        return sum(total_loss)/len(total_loss), sum(total_d1)/len(total_d1), sum(total_dn)/len(total_dn), sum(total_mse_list)/len(total_mse_list)


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_d1_list = []
            train_dn_list = []
            train_mse_list = []


            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
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
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        alpha, beta, gamma = self.args.alpha, self.args.beta, self.args.gamma

                        mse = criterion(outputs, batch_y)
                        d1 = delta_loss(outputs, batch_y, k=1)
                        dn = delta_loss(outputs, batch_y, k=self.args.delta_n)

                        loss = alpha * mse + beta * d1 + gamma * dn
                        train_loss.append(loss.item())

                        train_d1_list.append(d1.item())
                        train_dn_list.append(dn.item())
                        train_mse_list.append(mse.item())



                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    alpha, beta, gamma = self.args.alpha, self.args.beta, self.args.gamma

                    mse = criterion(outputs, batch_y)
                    d1 = delta_loss(outputs, batch_y, k=1)
                    dn = delta_loss(outputs, batch_y, k=self.args.delta_n)

                    loss = alpha * mse + beta * d1 + gamma * dn
                    train_loss.append(loss.item())

                    train_d1_list.append(d1.item())
                    train_dn_list.append(dn.item())
                    train_mse_list.append(mse.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_d1 = np.average(train_d1_list)
            train_dn = np.average(train_dn_list)
            train_mse = np.average(train_mse_list)


            vali_loss, vali_d1, vali_dn, vali_mse = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar('Train Loss', train_loss, epoch+1)
            self.writer.add_scalar('Valid Loss', vali_loss, epoch+1)

            self.writer.add_scalar('Train/MSE', train_mse, epoch + 1)
            self.writer.add_scalar('Train/Delta1', train_d1, epoch+1)
            self.writer.add_scalar(f'Train/Delta{self.args.delta_n}', train_dn, epoch+1)

            self.writer.add_scalar('Val/MSE', vali_mse, epoch + 1)
            self.writer.add_scalar('Val/Delta1', vali_d1, epoch + 1)
            self.writer.add_scalar(f'Val/Delta{self.args.delta_n}', vali_dn, epoch + 1)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
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
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return