import numpy as np

class StandardScaler:
    def fit_transform(self, series):
        self.mean = np.mean(series)
        self.std = np.std(series)
        return (series - self.mean) / self.std

    def inverse_transform(self, series):
        return series * self.std + self.mean