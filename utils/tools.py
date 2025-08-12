import os
import seaborn as sns
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from utils.metrics import MSE, MAE
from sklearn.manifold import TSNE
from torchdr import UMAP as torchdrUMAP
from config.config import DEFAULT_DATASET_SETTINGS
import glob

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == "None":
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, path, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, path, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        checkpoint = {
            'epoch' : epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, path + '/' + f"checkpoint_{epoch+1}_{val_loss:.6f}.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, pred=None, name='./pic/test.png', stock_str=None, date_labels=None, pred_len=None, is_reconstruction=False):
    """
    Results visualization
    """
    
    if (pred_len == 0) or (is_reconstruction == False):
        pred_mse = MSE(pred, true)
        pred_mae = MAE(pred, true)
    else:
        full_mse = MSE(pred, true)
        full_mae = MAE(pred, true)
        pred_mse = MSE(pred[-pred_len:], true[-pred_len:])
        pred_mae = MAE(pred[-pred_len:], true[-pred_len:])


    plt.figure()
    ax = plt.gca()

    if pred is not None:
        plt.plot(pred, label="Prediction", linewidth=2)

    plt.plot(true, label="GroundTruth", linewidth=2)

    if stock_str is not None and date_labels is not None:
        # title 설정
        if is_reconstruction and pred_len != 0:
            title = (
                f"< {stock_str} >   "
                f"Full MSE: {full_mse:.4f}, MAE: {full_mae:.4f} | "
                f"Pred MSE: {pred_mse:.4f}, MAE: {pred_mae:.4f}"
            )
        else:
            title = f"< {stock_str} >   MSE: {pred_mse:.4f}, MAE: {pred_mae:.4f}"
        plt.title(title, fontsize=11)
        plt.legend()

        # X축 날짜 표시
        step = max(1, len(date_labels)//10)
        tick_pos = list(range(0, len(date_labels), step))
        # tick_labels = [f"({i}) {date_labels[i]}" for i in tick_pos]
        tick_labels = [date_labels[i] if i < len(date_labels) else '' for i in tick_pos]

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)

    # 시각화 함수 안에서 axvline 추가
    if is_reconstruction and pred_len and pred_len < len(true):
        split_x = len(true) - pred_len
        ax.axvline(x=split_x, color='gray', linestyle='--', linewidth=1.5, label='Prediction Start')

    # x축 끝까지 보이게 설정
    ax.set_xlim(-1, max(len(true), len(pred)) + 4)
    plt.tight_layout()               
    plt.savefig(name, bbox_inches="tight")
    plt.show()

    return pred_mse


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def visual_attention_matrix(attn_matrix, save_root, suffix, model, dataset_name, model_id, target_feature, feature_names, freq, features, sample_idx=0):
    """
    Save the attention matrix as a heatmap to a structured path:
    {save_root}/{dataset_name}/{model_id}/{dataset_name}_{date}_{feature}_{idx}_attn.png
    """

    if features == 'S':
        feature_names = [target_feature]

    if freq in ['d', 'D']:
        date_feature_names = ['DayOfWeek', 'DayOfMonth', 'DayOfYear']
    elif freq in ['h', 'H']:
        date_feature_names = ['HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']
    elif freq in ['t', 'T']:
        date_feature_names = ['MinuteOfHour', 'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear']
    
    # 시간 정보 추가
    extended_feature_names = feature_names + date_feature_names

    assert attn_matrix.shape[0] == len(extended_feature_names), \
        f"Expected attention matrix shape ({len(extended_feature_names)}, {len(extended_feature_names)}), got {attn_matrix.shape}"
    
    # 경로 구성
    save_dir = os.path.join(save_root, model, f"{dataset_name}{suffix}", model_id)
    os.makedirs(save_dir, exist_ok=True)

    if features in ['MS', 'S'] and target_feature is not None:
        filename = f"{dataset_name}_{target_feature}_{sample_idx}_attn.png"
        title = f"< {dataset_name} - {target_feature} > Feature Attention Matrix_{sample_idx}"
    else:
        filename = f"{dataset_name}_M_{sample_idx}_attn.png"
        title = f"< {dataset_name} - M > Feature Attention Matrix_{sample_idx}"

    save_path = os.path.join(save_dir, filename)

    # 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap='viridis', xticklabels=extended_feature_names, yticklabels=extended_feature_names, annot=True, fmt=".2f")
    plt.title(title, fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# def save_embedding_tsv(vectors, flag=None, labels=None, filename_prefix='embedding', epoch=0,
#                        model='default_model', dataset='default_dataset', model_id='default_model_id'):
#     """
#     Save embedding vectors and metadata in TSV format for TensorFlow Embedding Projector.

#     Args:
#         vectors (np.ndarray): shape (N, D)
#         labels (list[str] or None): length N
#         filename_prefix (str): Prefix for filenames
#         epoch (int): Epoch number (used in filenames)
#         dataset (str): Dataset name, used to organize directories
#         model_id (str): Model identifier string
#         save_root (str): Root directory where files are saved
#     """
#     save_root="/data/pcw_workspace/Time-Series-Library/tsne_output"
    
#     # 생성할 디렉토리: /root/.../tsne_output/{dataset}/{model_id}/
#     save_dir = os.path.join(save_root, model, dataset, model_id, flag)
#     os.makedirs(save_dir, exist_ok=True)
    
#     if flag == 'test':
#         vector_path = os.path.join(save_dir, f"{flag}_{filename_prefix}_vectors.tsv")
#     else:
#         vector_path = os.path.join(save_dir, f"{flag}_{filename_prefix}_vectors_{epoch}epoch.tsv")

#     metadata_path = os.path.join(save_dir, f"{flag}_{filename_prefix}_metadata.tsv")

#     # 벡터 저장
#     np.savetxt(vector_path, vectors, delimiter="\t")

#     # 필요할때만 Metadata 저장
#     if not os.path.exists(metadata_path):
#         # 메타데이터 저장
#         np.savetxt(metadata_path, labels, delimiter="\t")
#         print(f"[TSV saved] {filename_prefix}_metadata.tsv")

#     if flag == 'test':
#         print(f"[TSV saved] {filename_prefix}_vectors.tsv")
#     else:
#         print(f"[TSV saved] {filename_prefix}_vectors_{epoch}epoch.tsv")

def visual_tsne(vectors, flag, save_path, filename_prefix, labels, epoch, model, dataset, model_id):
    """
    TSNE 시각화 및 PNG 저장 함수

    Args:
        vectors (np.ndarray or torch.Tensor): (N, D) 형태의 임베딩
        flag (str): "val" or "test"
        save_path (str): 저장 루트
        filename_prefix (str): 파일명 앞에 붙일 태그
        epoch (int or str): 에폭 번호
        model, dataset, model_id: 추가 정보 (파일명 or 로그용)
    """
    
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    if not isinstance(vectors, np.ndarray):
        raise TypeError("vectors must be np.ndarray or torch.Tensor")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be np.ndarray or torch.Tensor")
    
    # 라벨 정의
    targets = DEFAULT_DATASET_SETTINGS[dataset]['targets']
    mark_in = DEFAULT_DATASET_SETTINGS[dataset]['mark_in']
    label_names = targets + mark_in
    num_classes = len(np.unique(labels))

    # 차원 축소 시도 : UMAP -> TSNE
    try:
        x_tensor = torch.tensor(vectors, dtype=torch.float32, device='cuda')
        umap = torchdrUMAP(
            n_neighbors = 7,
            n_components = 2,
            compile = False,
            device ='cuda',
            max_iter=1200,
            min_dist=0.1,
            spread=1.5
                        )
        reduced = umap.fit_transform(x_tensor).detach().cpu().numpy()
        method_used = "UMAP"
    except RuntimeError as e:
        print(f"[WARN] UMAP failed: {str(e)}. Falling back to TSNE...")
        tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=30)
        reduced = tsne.fit_transform(vectors)
        method_used = "TSNE"

    # 색상 지정
    color_palette = sns.color_palette("husl", num_classes)
    label_to_color = {label: color_palette[i] for i, label in enumerate(np.unique(labels))}
    label_to_name = {i: label_names[i] for i in sorted(np.unique(labels))}

    # 점 크기 조절 조건
    if filename_prefix.startswith("position_embedding"):
        dot_size = 15
    else:
        dot_size = 1

    # 시각화
    plt.figure(figsize=(10, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(
            reduced[idx, 0], reduced[idx, 1],
            label=label_to_name[label],
            s=dot_size,
            alpha=0.5,
            color=label_to_color[label]
        )
    
    # title
    if flag == "val":
        plt.title(f"{filename_prefix} | {flag} _ epoch {epoch}")
    elif flag == "test":
        plt.title(f"{filename_prefix} | {flag}")
    plt.xlabel(f"{method_used}-1")
    plt.ylabel(f"{method_used}-2")
    plt.grid(True)

    # 범례 그림 바깥으로
    legend = plt.legend(
        title="Feature",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0
    )

    # 범례 설정
    for handle in legend.legendHandles:
        handle.set_sizes([35])
        handle.set_alpha(0.7)

    # 레이아웃
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 오른쪽 여백 확보

    # 저장 경로
    if flag == "val":
        filename = f"{filename_prefix}_{flag}_epoch{epoch}_{method_used}.png"
    elif flag == "test":
        filename = f"{filename_prefix}_{flag}_{method_used}.png"

    save_full_path = os.path.join(save_path, flag, filename)
    os.makedirs(os.path.dirname(save_full_path), exist_ok=True)

    plt.savefig(save_full_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved {method_used} plot to: {save_full_path}")