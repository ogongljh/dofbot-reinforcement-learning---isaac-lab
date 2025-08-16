'''
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
        '''

from http.server import BaseHTTPRequestHandler, HTTPServer

class HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)  # HTTP 200 OK
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>HELLO WORLD</h1></body></html>")

if __name__ == "__main__":
    host = "0.0.0.0"  # 모든 네트워크 인터페이스에서 수신
    port = 8888
    server = HTTPServer((host, port), HelloHandler)
    print(f"서버 실행 중: http://{host}:{port}")
    server.serve_forever()
