import serial
import pickle
import numpy as np
import mysql.connector

# ✅ 시리얼 포트 설정
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

# ✅ 저장된 모델 불러오기
with open("best_ann_model.pkl", "rb") as f:
    W1, W2, emotion_mapping, mean, scale = pickle.load(f)

# ✅ 개선된 Softmax 함수
def scaled_softmax(x, scale_factor=0.1):#한쪽 값만 크게 예측하지 않도록 *0.1함
    x = x * scale_factor
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ✅ 순전파 (Forward Propagation)
def forward(X, W1, W2):
    X_bias = np.column_stack((X, np.ones(len(X))))
    Z1 = np.dot(X_bias, W1)
    A1 = 1 / (1 + np.exp(-Z1))
    H_bias = np.column_stack((A1, np.ones(len(A1))))
    Z2 = np.dot(H_bias, W2)
    A2 = scaled_softmax(Z2)
    return A2, Z2

previous_gsr = None
print("✅ 시리얼 포트 연결 완료. 데이터 수신 중...")
try:
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            data = line.split("\t")

            if len(data) == 4:
                # ✅ 측정값 읽기
                heart_rate = int(data[0])
                spo2 = int(data[1])
                temperature = float(data[2])
                gsr = int(data[3])

                # ✅ gsr_diff 계산
                if previous_gsr is None:
                    gsr_diff = 0
                else:
                    gsr_diff = gsr - previous_gsr
                previous_gsr = gsr

                # ✅ 입력 데이터 전처리 및 예측
                input_data = np.array([[heart_rate, gsr, gsr_diff]])
                input_data = (input_data - mean) / scale

                predictions, Z2 = forward(input_data, W1, W2)#순전파 작동
                probabilities = np.round(predictions[0] * 100, 2)#각 확률
                predicted_label = np.argmax(predictions)#제일 큰 값의 번호
                predicted_emotion = emotion_mapping[predicted_label]#라벨링된 해당 감정

                # ✅ 예측 결과 표시
                prob_str = ", ".join([f"{emotion_mapping[label]}: {probabilities[label]}%" for label in range(len(probabilities))])
                print(f"\n📊 실시간 측정값 및 예측 결과")
                print(f"------------------------------------")
                print(f"심박수: {heart_rate} | 산소포화도: {spo2} | 온도: {temperature} | GSR: {gsr} | GSR 변화량: {gsr_diff}")
                print(f"예측된 감정: {predicted_emotion} ({probabilities[predicted_label]}%)")
                print(f"확률 분포: {prob_str}")
                print(f"------------------------------------")

                # ✅ 예측 결과를 데이터베이스에 저장
                query = """
                INSERT INTO prediction_data 
                (heart_rate, spo2, temperature, gsr, gsr_diff, predicted_emotion, probability)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                values = (heart_rate, spo2, temperature, gsr, gsr_diff, predicted_emotion, probabilities[predicted_label])
                cursor.execute(query, values)
                db.commit()

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