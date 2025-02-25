import torch
import polars as pl
import numpy as np


def df_mse(df: pl.DataFrame, y_col: str, pred_col: str) -> float:
    criterion = torch.nn.MSELoss()
    y_true = torch.tensor(df[y_col].to_list(), dtype=torch.float32)
    y_pred = torch.tensor(df[pred_col].to_list(), dtype=torch.float32)
    loss = criterion(y_true, y_pred)
    return loss.item()


def ndarray_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    criterion = torch.nn.MSELoss()
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    loss = criterion(y_true, y_pred)
    return loss.item()
