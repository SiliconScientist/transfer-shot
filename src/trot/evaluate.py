import torch
import numpy as np


def get_rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    criterion = torch.nn.MSELoss()
    y_true = torch.tensor(y, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32).squeeze()
    rmse = float(np.sqrt(criterion(y_true, y_pred)))
    return rmse
