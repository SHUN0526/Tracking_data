import mysql.connector
import serial
import time

SERIAL_PORT = "COM12"
BAUD_RATE = 115200

#MySQL 데이터베이스 연결
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0101",
    database="sensor_db"
)

cursor = db.cursor() #SQL데이터베이스 상호작용 커서
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)#시리얼 포트와 통신속도도 

print("✅ MySQL 연결 완료. 시리얼 포트에서 데이터 수신 중...")

# 🔥 연속적인 -1 값 감지 시 오류 메시지 출력
error_count = 0
max_error_count = 10  # 10번 연속 -1이 나오면 경고 출력
try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            #시리얼 한 줄씩 읽기
            #시리얼 통신으로 들어온 데이터는 바이트형식이어서 텍스트로 변환-디코드
            #앞 뒤 공백 및 줄바꿈 문자 제거 -strip()
            data = line.split("\t")
            #탭 만큼 구분

            if len(data) == 4: #심박수, 산소포화도, 온도, 피부전도도
                heart_rate = int(data[0])
                spo2 = int(data[1])
                temperature = float(data[2])
                gsr = int(data[3])

                #오류 감지 (-1 값이 연속 10번 나오면 경고 출력) - 이 이프문 지나가고 저장하기에 필요
                if heart_rate == -1 or spo2 == -1 or temperature == -1 or gsr == -1:
                    error_count += 1
                    print(f"⚠️ 오류 감지: -1 값 {error_count}회 발생")

                    if error_count >= max_error_count:
                        print("🚨 센서 오류 지속 발생! 점검 필요.")
                    continue  # -1 값이 나오면 저장하지 않음
                
                # ✅ 정상 데이터 저장
                query = "INSERT INTO sensor_data (heart_rate, spo2, temperature, gsr) VALUES (%s, %s, %s, %s)"
                #쿼리(데이터베이스에게 요청)(heart_rate, spo2, temperature, gsr)는 열이름 
                values = (heart_rate, spo2, temperature, gsr)
                #실제 넣을 값들
                cursor.execute(query, values)
                #쿼리와 값들을 데이터베이스에 전달
                db.commit()
                #저장

                # ✅ 정상 데이터 수신 시 error_count 초기화
                error_count = 0  
                print(f"📡 저장됨: 심박수={heart_rate}, 산소포화도={spo2}, 온도={temperature}, GSR={gsr}")
        
        except Exception as e:
            print("❌ 오류 발생:", e)

except KeyboardInterrupt:
    print("\n🛑 프로그램이 강제로 종료되었습니다. (Ctrl + C)")

finally:
    # ✅ 프로그램 종료 전에 안전하게 닫기
    if ser.is_open:
        ser.close()
    if db.is_connected():
        db.close()
    print("✅ 시리얼 포트와 데이터베이스 연결이 안전하게 종료되었습니다.")