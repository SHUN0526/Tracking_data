import mysql.connector
import serial
import time

# 시리얼 포트 설정 (사용 중인 COM 포트로 변경)
SERIAL_PORT = "COM12"  # 실제 Arduino가 연결된 포트
BAUD_RATE = 115200

# MySQL 데이터베이스 연결
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0101",  # MySQL 비밀번호
    database="sensor_db"
)

cursor = db.cursor()
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

print("✅ MySQL 연결 완료. 시리얼 포트에서 데이터 수신 중...")

while True:
    try:
        # 시리얼 포트에서 한 줄 읽기
        line = ser.readline().decode('utf-8').strip()
        data = line.split("\t")

        if len(data) == 4:  # 데이터 개수가 4개인지 확인
            heart_rate = int(data[0])
            spo2 = int(data[1])
            temperature = float(data[2])
            gsr = int(data[3])

            # MySQL에 데이터 삽입
            query = "INSERT INTO sensor_data (heart_rate, spo2, temperature, gsr) VALUES (%s, %s, %s, %s)"
            values = (heart_rate, spo2, temperature, gsr)
            cursor.execute(query, values)
            db.commit()

            print(f"📡 저장됨: 심박수={heart_rate}, 산소포화도={spo2}, 온도={temperature}, GSR={gsr}")

    except Exception as e:
        print("❌ 오류 발생:", e)

ser.close()
db.close()
