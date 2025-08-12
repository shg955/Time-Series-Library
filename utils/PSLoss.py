import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

class PSLoss:
    def __init__(self, model, patch_len_threshold=24):
            self.model = model
            self.patch_len_threshold = patch_len_threshold
            self.kl_loss = nn.KLDivLoss(reduction='none')

    def debug_tensor(self, name, tensor):
        if torch.isnan(tensor).any():
            print(f"[NaN DETECTED] {name} contains NaNs")
            print(f"→ Shape: {tensor.shape}")
            print(f"→ Values: {tensor}")

        elif torch.isinf(tensor).any():
            print(f"[INF DETECTED] {name} contains Infs")
            print(f"→ Shape: {tensor.shape}")
            print(f"→ Values: {tensor}")


    def create_patches(self, x, patch_len, stride):
        
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(self, true, pred):
        # Get patch length an stride
        true_fft = torch.fft.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)

        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = max(1, patch_len // 2)

        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch

    def patch_wise_structural_loss(self, true_patch, pred_patch):
        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        
        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var + 1e-6)
        pred_patch_std = torch.sqrt(pred_patch_var + 1e-6)
        
        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        
        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        
        # print(f"[DEBUG] patch_linear_corr mean: {patch_linear_corr.mean().item()}")
        # print(f"[DEBUG] true_patch_std mean: {true_patch_std.mean().item()}, pred_patch_std mean: {pred_patch_std.mean().item()}")
        # print(f"[DEBUG] true_pred_patch_cov mean: {true_pred_patch_cov.mean().item()}")
        # [DEBUG] patch_linear_corr mean: 1.0
        # [DEBUG] true_patch_std mean: 0.0, pred_patch_std mean: 0.0017963189166039228
        # [DEBUG] true_pred_patch_cov mean: 0.0

        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()

        # PSLoss nan 원인 부분 시각화
        # if (
        #     patch_linear_corr.mean().item() == 1.0 and
        #     true_pred_patch_cov.mean().item() == 0.0 and
        #     true_patch_std.mean().item() == 0.0
        # ):
        #     import matplotlib.pyplot as plt
        #     import os
        #     import numpy as np

        #     os.makedirs("debug_patches", exist_ok=True)
        #     B, C, N, L = true_patch.shape
        #     for b in range(B):
        #         for c in range(C):
        #             for n in range(N):
        #                 patch = true_patch[b, c, n].detach().cpu().numpy()
        #                 plt.figure(figsize=(8, 2))
        #                 plt.plot(patch)
        #                 plt.title(f"True Patch b{b}_c{c}_n{n} | Mean: {patch.mean():.4f}, Var: {patch.var():.4f}")
        #                 plt.tight_layout()
        #                 plt.grid(True)
        #                 plt.savefig(f"debug_patches/true_patch_b{b}_c{c}_n{n}.png")
        #                 plt.close()
        #                 print(f"Saved suspicious patch")
        
        return linear_corr_loss, var_loss, mean_loss

    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
        
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        # Gradiant based dynamic weighting
        corr_gradient = torch.autograd.grad(corr_loss, self.model.projection.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.model.projection.parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.model.projection.parameters(), create_graph=True)[0]
        
        self.debug_tensor("corr_gradient", corr_gradient)
        self.debug_tensor("var_gradient", var_gradient)
        self.debug_tensor("mean_gradient", mean_gradient)

        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        alpha = gradiant_avg.norm().detach() / (corr_gradient.norm().detach() + 1e-8)
        beta =  gradiant_avg.norm().detach() /  (var_gradient.norm().detach() + 1e-8)
        gamma = gradiant_avg.norm().detach() / (mean_gradient.norm().detach() + 1e-8)
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()

        self.debug_tensor("alpha_norm", alpha)
        self.debug_tensor("beta_norm", beta)
        self.debug_tensor("gamma_norm", gamma)
        
        return alpha, beta, gamma
    
    def compute_loss(self, true, pred):

        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)
        
        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)

        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)

        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss
        
        return ps_loss, alpha, corr_loss, beta, var_loss, gamma, mean_loss