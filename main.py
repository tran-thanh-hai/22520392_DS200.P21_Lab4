import socket
import numpy as np
from models.arima import ARIMAModel
from transforms.normalize import StandardScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType

HOST = 'localhost'
PORT = 6100

def receive_batch(conn, is_train=True):
    batch_size_line = b""
    while not batch_size_line.endswith(b"\n"):
        batch_size_line += conn.recv(1)
    batch_size = int(batch_size_line.decode().strip())
    batch = []
    for _ in range(batch_size):
        line = b""
        while not line.endswith(b"\n"):
            line += conn.recv(1)
        splitted = line.decode().strip().split(',')
        if is_train:
            value = float(splitted[2])  # Vehicles
        else:
            value = float(splitted[0])  # ID
        batch.append(value)
    return batch

def main():
    spark = SparkSession.builder.appName("ARIMA_Streaming").getOrCreate()

    schema = StructType([StructField("value", DoubleType(), True)])

    spark_df = spark.createDataFrame([], schema)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print("Đã kết nối tới server.")

    peek = s.recv(4, socket.MSG_PEEK)
    if peek.startswith(b"END"):
        s.recv(4)
        print("Không có dữ liệu.")
        return

    batch_size_line = b""
    while not batch_size_line.endswith(b"\n"):
        batch_size_line += s.recv(1)
    batch_size = int(batch_size_line.decode().strip())
    first_batch = []
    for _ in range(batch_size):
        line = b""
        while not line.endswith(b"\n"):
            line += s.recv(1)
        splitted = line.decode().strip().split(',')
        first_batch.append(splitted)
    is_train = len(first_batch[0]) >= 4

    batch = []
    for splitted in first_batch:
        if is_train:
            value = float(splitted[2])
        else:
            value = float(splitted[0])
        batch.append(value)

    batch_df = spark.createDataFrame([(float(x),) for x in batch], schema)
    spark_df = spark_df.union(batch_df)

    s.sendall(b"OK\n")
    pbar = tqdm(desc="Nhận và train batch", unit="batch")
    pbar.update(1)

    while True:
        peek = s.recv(4, socket.MSG_PEEK)
        if peek.startswith(b"END"):
            s.recv(4)
            break
        batch = receive_batch(s, is_train)
        if not batch:
            break

        batch_df = spark.createDataFrame([(float(x),) for x in batch], schema)
        spark_df = spark_df.union(batch_df)

        series = np.array([row['value'] for row in spark_df.collect()])
        scaler = StandardScaler()
        series_scaled = scaler.fit_transform(series)
        model = ARIMAModel(order=(1,1,1))
        model.train(series_scaled)
        preds_scaled = model.predict(steps=len(series_scaled))
        preds = scaler.inverse_transform(preds_scaled)
        mae = mean_absolute_error(series, preds)
        s.sendall(b"OK\n")
        pbar.update(1)
    pbar.close()
    print(f"Đánh giá MAE: {mae:.4f} trên bộ dữ liệu gồm {len(series)} dòng")

    s.close()
    spark.stop()
    print("Đã nhận xong và train xong.")

if __name__ == "__main__":
    main()