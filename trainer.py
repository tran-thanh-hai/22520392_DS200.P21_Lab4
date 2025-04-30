import socket
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from models.arima import ARIMAModel
from transforms.normalize import StandardScaler
from sklearn.metrics import mean_absolute_error

class Trainer:
    def __init__(self, host, port, order=(1,1,1)):
        self.host = host
        self.port = port
        self.order = order
        self.data = []

    def receive_data(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        buffer = ""
        while True:
            msg = s.recv(1024)
            if not msg:
                break
            buffer += msg.decode()
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                value = float(line.split(',')[0])
                self.data.append(value)
        s.close()

    def train_and_evaluate(self):
        spark = SparkSession.builder.appName("ARIMA_Trainer").getOrCreate()

        schema = StructType([StructField("value", DoubleType(), True)])
        df = spark.createDataFrame([(float(x),) for x in self.data], schema)
        df.show(5)

        series = np.array(self.data)
        scaler = StandardScaler()
        series_scaled = scaler.fit_transform(series)

        model = ARIMAModel(order=self.order)
        model.train(series_scaled)
        preds_scaled = model.predict(steps=len(series_scaled))
        preds = scaler.inverse_transform(preds_scaled)

        mae = mean_absolute_error(series, preds)
        print(f"MAE: {mae}")

        spark.stop()

if __name__ == "__main__":
    trainer = Trainer(host='localhost', port=6100)
    print("Đang nhận dữ liệu từ server gửi...")
    trainer.receive_data()
    print("Huấn luyện và đánh giá mô hình ARIMA...")
    trainer.train_and_evaluate()