import serial
import pickle
import numpy as np

# ✅ 저장된 ANN 모델 불러오기
with open("ann_model.pkl", "rb") as f:
    W1, W2 = pickle.load(f)

# ✅ 순전파 (Forward Propagation) - 감정 예측
def forward(X, W1, W2):
    X_bias = np.column_stack((X, np.ones(len(X))))  # 바이어스 추가
    Z1 = np.dot(X_bias, W1)  # 입력층 → 은닉층
    A1 = 1 / (1 + np.exp(-Z1))  # 활성화 함수 적용
    H_bias = np.column_stack((A1, np.ones(len(A1))))  # 은닉층 바이어스 추가
    Z2 = np.dot(H_bias, W2)  # 은닉층 → 출력층
    A2 = 1 / (1 + np.exp(-Z2))  # 활성화 함수 적용
    return A2

# ✅ 시리얼 포트 설정 (ESP32 등과 연결)
ser = serial.Serial("COM12", 115200)

while True:
    try:
        # ✅ 데이터 읽기 & 공백 제거
        line = ser.readline().decode('utf-8').strip()

        # ✅ "\t"(탭) 또는 " "(공백)으로 데이터 분할
        values = line.split("\t") if "\t" in line else line.split()

        # ✅ 데이터 개수가 4개 이상인지 확인 (0번: 심박수, 3번: GSR)
        if len(values) < 4:
            print(f"⚠️ 잘못된 데이터 수신 (필요한 값 부족): {line}")
            continue
        
        # ✅ 데이터 변환 (심박수 & GSR)
        heart_rate = float(values[0])  # 0번 인덱스가 심박수
        gsr = float(values[3])  # 3번 인덱스가 GSR 값

        # ✅ GSR 변화량 계산 (처음엔 0)
        gsr_diff = 0  

        # ✅ 감정 예측 실행
        input_data = np.array([[heart_rate, gsr, gsr_diff]])
        emotion_pred = forward(input_data, W1, W2)

        # ✅ 예측된 감정 출력
        emotion_label = "기쁨" if np.round(emotion_pred) == 1 else "긴장"
        print(f"🎯 실시간 감정 예측: {emotion_label}")

    except ValueError as ve:
        print(f"❌ 데이터 변환 오류: {ve} → 수신된 데이터: {line}")
    except Exception as e:
        print(f"❌ 기타 오류 발생: {e}")
