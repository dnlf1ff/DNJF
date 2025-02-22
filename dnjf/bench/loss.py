#OOP
from ase.io import read
import numpy as np
from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
        median_absolute_error,
        explained_variance_score
)

from util import load_dict, set_env, save_dict,tot_sys_mlps, dishwash

def get_nions():
    from ase.io import read
    for system in systems:
        atoms = read(os.path.join(os.environ['BARK'],'TM23',f'{system}.xyz'))
        errors[system]['nions'] = len(atoms)
    save_dict(errors, 'errors')


class ErrorRecorder:
    def __init__(self):
        set_env('bench')
        self.errors = {}
        systems, mlps = tot_sys_mlps('tot') 
        self.systems = systems
        self.mlps = mlps
        self.loss_metrics = ['mae','rmse','r2','mape','mav','evs']
        self.labels = ['pe','force', 'stress', 'vol']
        #TODO: make to keys
        self.system_out = {}
        for system in self.systems:
            self.system_out[system] = self._load_dict(system)
            self.errors[system] = {}
            for mlp in self.mlps:
                self.errors[system][mlp] = {}
                for label in self.labels:
                    self.errors[system][mlp][label] = {}
                    for loss_metric in self.loss_metrics:
                        self.errors[system][mlp][label][loss_metric] = None

    def _nions(self):
        for system in self.systems:
            atoms = read(os.path.join(os.environ['BARK'],'TM23',f'{system}.xyz')) 
            self.errors[system]['nions'] = len(atoms)

    def _load_dict(self, system):
        return load_dict(system)

    def _save_dict(self):
        save_dict(data=self.errors, path='errors')

    def _get_y(self, system, mlp, label, dft=False):
        if dft:
            _key = f'{label}-dft'
        else:
            _key = f'{label}-{mlp}'
        _y = self.system_out[system][_key]
        return  _y

    @staticmethod
    def _flatten(out, label):
        if label in ['force','stress']:
            _out = np.concatenate([f for f in out])
        else:
            _out = np.asarray(out)
        return _out

    @staticmethod
    def percentage_error(y_true, y_pred):
        percent_error = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return percent_error

    @staticmethod
    def _average(_y_pred):
        _averaged_y = [np.mean([np.abs(_y) for _y in _y_pred])]
        return _averaged_y

    def mean_absolute_value(self, y_pred, label):
        if label in ['force','stress']:
            y_pred = self._average(y_pred)
            return np.mean(y_pred)
        else:
            return np.mean(np.abs(y_pred))

    def _calc(self, y_true, y_pred, loss_function):
        if 'mae' in loss_function.lower() or loss_function == 'Mean Absolute Error':
            return mean_absolute_error(y_true, y_pred)
        elif 'mse' in loss_function.lower() or loss_function == 'Mean Squared Error':
            return mean_squared_error(y_true, y_pred)
        elif 'r2' in loss_function.lower() or loss_function == 'R2 score':
            return r2_score(y_true, y_pred)
        elif 'mape' in loss_function.lower() or loss_function == 'Mean Absolute Percentage Error':
            return mean_absolute_percentage_error(y_true, y_pred)
        elif 'evs' in loss_function.lower() or loss_function == 'Explained Variance Score':
            return explained_variance_score(y_true, y_pred)
        elif 'per' in loss_function.lower() or loss_function == 'Percentage Errorr':
            return self.percentage_error(y_true, y_pred)

    def record(self, y_true, y_pred, system, mlp, label):
        y_true = self._flatten(y_true, label)
        y_pred = self._flatten(y_pred, label)
        for loss_metric in self.loss_metrics:
            self.errors[system][mlp][label][loss_metric]=self._calc(y_true, y_pred, loss_metric)
        self.errors[system][mlp][label]['mav'] = self.mean_absolute_value(y_pred, label)

    def run(self):
        for system in self.systems:
            for mlp in self.mlps:
                for label in self.labels:
                    y_true = self._get_y(system, mlp, label, dft=True)
                    y_pred = self._get_y(system, mlp, label)
                    self.record(y_true, y_pred, system, mlp, label)
            print(f'{system} done')
            self._save_dict()

def main():
    error_recorder = ErrorRecorder()
    error_recorder.run()

if __name__ == '__main__':
    main()
