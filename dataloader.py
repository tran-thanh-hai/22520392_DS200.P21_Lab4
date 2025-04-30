import pandas as pd

class DataLoader:
    def __init__(self, spark, file_path):
        self.spark = spark
        self.file_path = file_path

    def load_series(self):
        df = pd.read_csv(self.file_path)
        if 'ID' in df.columns:
            return df['ID'].values
        else:
            raise Exception("Không tìm thấy cột dữ liệu phù hợp!")

    def load_spark_df(self):
        return self.spark.read.csv(self.file_path, header=True, inferSchema=True)