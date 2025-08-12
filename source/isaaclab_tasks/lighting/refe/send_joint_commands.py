import socket
import pickle
import time

IP = '192.168.1.213'  # Jetson Nano의 wlan0 IP
PORT = 50007

angles_deg = [90, 45, 60, 90, 90, 90]  # 6개 조인트 각도 (degree)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((IP, PORT))
    print("🟢 서버 연결됨")
    while True:
        s.sendall(pickle.dumps(angles_deg))
        print("📤 전송 완료:", angles_deg)
        time.sleep(2)