from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def train(self, series):
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model

    def predict(self, steps=1):
        if self.fitted_model is None:
            raise Exception("Model chưa được train!")
        return self.fitted_model.forecast(steps=steps)