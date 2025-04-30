import socket
import pandas as pd
import time
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', required=True)
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', type=int, default=6100)
parser.add_argument('--interval', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

df = pd.read_csv(args.file)
total = len(df)
batch_size = args.batch_size

is_train = 'Vehicles' in df.columns

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((args.host, args.port))
s.listen(1)
print(f"Đang chờ client kết nối tại {args.host}:{args.port} ...")
conn, addr = s.accept()
print(f"Đã kết nối với {addr}")

try:
    for i in tqdm(range(0, total, batch_size), desc="Đang gửi dữ liệu"):
        batch = df.iloc[i:i+batch_size]

        conn.sendall(f"{len(batch)}\n".encode())

        for _, row in batch.iterrows():
            msg = ','.join(map(str, row.values)) + '\n'
            conn.sendall(msg.encode())
            time.sleep(args.interval)

        ack = conn.recv(16)
        if ack.decode().strip() != "OK":
            print("Client không xác nhận, dừng gửi.")
            break

    conn.sendall(b"END\n")
except Exception as e:
    print("Lỗi khi gửi:", e)
finally:
    conn.close()
    print("Đã đóng kết nối.")