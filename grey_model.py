
import numpy as np


class GreyModel:
    """GM(1, 1) model for time series prediction."""

    def __init__(self, data):
        """
        :param data: data should be a list or 1D ndarray
        """
        if isinstance(data, list):
            data = np.array(data)
        assert len(data.shape) == 1 or data.shape[1] == 1, 'input array should be 1D'
        self.data = data
        self.flag = -1

    def level_check(self):
        """apply level check on input data."""
        n = len(self.data)
        level = self.data[:-1] / self.data[1:]
        lower_bound, upper_bound = np.exp(-2/(n+2)), np.exp(2/(n+2))
        if np.logical_and(level > lower_bound, level < upper_bound).all():
            self.flag = 1
        else:
            print('level check failed')
            self.flag = 0

    def fit(self, return_func=False):
        if self.flag == -1:
            self.level_check()
        elif self.flag == 0:
            raise ValueError('pass level check first')
        X_1 = self.data.copy()
        np.cumsum(X_1, out=X_1)
        Z = np.zeros(X_1.shape[0])
        Z[0] = X_1[0]
        for i in range(1, X_1.shape[0]):
            Z[i] = 0.5 * (X_1[i] + X_1[i - 1])
        mat = np.vstack((-Z, np.ones(len(Z)))).T
        a, b = np.linalg.lstsq(mat, X_1, rcond=None)[0]
        self._func = lambda x: (self.data[0] - b/a)*(1 - np.exp(a))*np.exp(-a*(x-1))

    def predict(self, x):
        """x should be a list or 1D ndarray"""
        return list(map(self._func, x))


if __name__ == '__main__':
    test_data = np.array([33.27, 43.41, 62.06, 101.72, 131.15, 170.73, 217.69, 296.39, 440.69, 457.12, 559.11])
    grey = GreyModel(test_data)
    grey.fit(return_func=True)
