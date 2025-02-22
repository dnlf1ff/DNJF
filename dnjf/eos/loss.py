#OOP
import numpy as np
from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
        median_absolute_error,
        explained_variance_score
)

from util import load_dict, set_env, save_dict,tot_sys_mlps

def _flatten(out):
    _out = np.concatenate([f for f in out])
    return _out

def percentage_error(y_true, y_pred):
    percent_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return percent_error

def _loss(y_true, y_pred, loss_function):
    if 'mae' in loss_function.lower() or loss_function == 'Mean Absolute Error':
        return mean_absolute_error(y_true, y_pred)
    elif 'mse' in loss_function.lower() or loss_function == 'Mean Squared Error':
        return mean_squared_error(y_true, y_pred)
    elif 'r2' in loss_function.lower() or loss_function == 'R2 score':
        return r2_score(y_true, y_pred)
    elif 'mape' in loss_function.lower() or loss_function == 'Mean Absolute Percentage Error':
        return mean_absolute_percentage_error(y_true, y_pred)
    elif 'mav' in loss_function.lower() or loss_function == 'Median Absolute Error':
        return median_absolute_error(y_true, y_pred)
    elif 'evs' in loss_function.lower() or loss_function == 'Explained Variance Score':
        return explained_variance_score(y_true, y_pred)
    elif 'per' in loss_function.lower() or loss_function == 'Percentage Errorr':
        return mean_squared_error(y_true, y_pred, squared=False)
    else:
        raise ValueError('Invalid loss function')


def main():
    set_env('
