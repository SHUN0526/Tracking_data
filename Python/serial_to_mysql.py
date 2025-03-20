import mysql.connector
import serial
import time

# ✅ 시림 포트 설정 (uc0ac용 중인 COM 포트로 변경)
SERIAL_PORT = "COM12"
BAUD_RATE = 115200

# ✅ MySQL 데이터베이스 연결
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0101",
    database="sensor_db"
)

cursor = db.cursor()
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

print("✅ MySQL 연결 완료. 시리얼 포트에서 데이터 수신 중...")

# 🔥 연속적인 -1 값 감지 시 오류 메시지 출력
error_count = 0
max_error_count = 10  # 10번 연속 -1이 나오면 경고 출력

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        data = line.split("\t")

        if len(data) == 4:
            heart_rate = int(data[0])
            spo2 = int(data[1])
            temperature = float(data[2])
            gsr = int(data[3])

            # ✅ 오류 감지 (-1 값이 연속 10번 나오면 경고 출력)
            if heart_rate == -1 or spo2 == -1 or temperature == -1 or gsr == -1:
                error_count += 1
                print(f"⚠️ 오류 감지: -1 값 {error_count}회 발생")

                if error_count >= max_error_count:
                    print("🚨 센서 오류 지속 발생! 점검 필요.")
                continue  # -1 값이 나오면 저장하지 않음
            
            # ✅ 정상 데이터 저장
            query = "INSERT INTO sensor_data (heart_rate, spo2, temperature, gsr) VALUES (%s, %s, %s, %s)"
            values = (heart_rate, spo2, temperature, gsr)
            cursor.execute(query, values)
            db.commit()

            # ✅ 정상 데이터 수신 시 error_count 초기화
            error_count = 0  
            print(f"📡 저장됨: 심박수={heart_rate}, 산소포화도={spo2}, 온도={temperature}, GSR={gsr}")

    except Exception as e:
        print("❌ 오류 발생:", e)

ser.close()
db.close()
