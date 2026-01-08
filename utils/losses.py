# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb

from torchmetrics.regression import MeanSquaredLogError, RelativeSquaredError, LogCoshError
from pytorch_forecasting.metrics import QuantileLoss
import torch.nn.functional as F


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class mbe_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, forecast, target):
        return t.mean(target - forecast)
    

class rae_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, forecast, target):
        return t.sum(t.abs(target - forecast)) / (t.sum(t.abs(target - t.mean(target))) + 1e-10)


class rse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rse = RelativeSquaredError()
        
    def forward(self, forecast, target):
        return self.rse(forecast, target)
    

class rmse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, forecast, target):
        return t.sqrt(self.mse(forecast, target))
    

class msle_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.msle = MeanSquaredLogError()
        
    def forward(self, forecast, target):
        return self.msle(forecast, target)
    

class rmsle_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, forecast, target):
        return t.sqrt(t.mean(t.square(t.log1p(forecast) - t.log1p(target))))
    

class nrmse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = rmse_loss()
        
    def forward(self, forecast, target):
        return self.rmse(target, forecast) / (target.max() - target.min())
    

class rrmse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, forecast, target):
        n = len(target)
        num = t.sum(t.square(target - forecast)) / n
        den = t.sum(t.square(forecast))
        squared_error = num/den
        rrmse = t.sqrt(squared_error)
        return rrmse


class log_cosh_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logcosh = LogCoshError()
        
    def forward(self, forecast, target):
        return self.logcosh(forecast, target)
    

class quantile_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantlie = QuantileLoss()
        
    def forward(self, forecast, target):
        return self.quantlie(forecast, target)


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: t.Tensor) -> t.Tensor:
        # x: [B, T, C]
        batch_size, seq_len, channels = x.shape
        device = x.device
        dtype = x.dtype

        # 지수 가중치 생성 (float64로 계산 후 나중에 캐스팅)
        powers = t.flip(
            t.arange(seq_len, dtype=t.float64, device=device),
            dims=(0,),
        )  # [seq_len]
        weights = t.pow((1 - self.alpha), powers)  # [seq_len]
        divisor = weights.clone()

        # EMA용 가중치 조정: 0번째는 그대로, 1:부터 α 곱
        if seq_len > 1:
            weights[1:] = weights[1:] * self.alpha

        # 배치/채널 브로드캐스트를 위한 reshape
        weights = weights.view(1, seq_len, 1) 
        divisor = divisor.view(1, seq_len, 1) 

        # EMA 계산 (더 안정적인 float64로 계산)
        x_fp64 = x.to(t.float64)
        ema = t.cumsum(x_fp64 * weights, dim=1)
        ema = ema / divisor

        return ema.to(dtype)



class DECOMP(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, alpha):
        super(DECOMP, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x):
        moving_average = self.ma(x)   # trend
        res = x - moving_average      # seasonality
        return res, moving_average


class DBLoss(nn.Module):
    """Decomposition-based loss (trend + season loss)"""

    def __init__(self, alpha, beta):
        super().__init__()
        self.decomp = DECOMP(alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")  # season
        self.mae = nn.L1Loss(reduction="mean")   # trend

    def forward(self, pred, target):
        # pred/target: [B, T, C]
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)

        # scale alignment
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()

        return self.beta * season_loss + (1 - self.beta) * trend_loss
