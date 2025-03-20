import serial
import pickle
import numpy as np

# ✅ 저장된 ANN 모델 불러오기
with open("best_ann_model.pkl", "rb") as f:
    W1, W2, emotion_mapping, mean, scale = pickle.load(f)

print(f"📢 감정 라벨 매핑 확인: {emotion_mapping}")

# ✅ 소프트맥스 함수 (안정화 적용)
def softmax(x):
    x = x - np.max(x)  # 오버플로우 방지
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# ✅ 순전파 (Forward Propagation)
def forward(X, W1, W2):
    X_bias = np.column_stack((X, np.ones(len(X))))  # 바이어스 추가
    Z1 = np.dot(X_bias, W1)
    A1 = 1 / (1 + np.exp(-Z1))  

    H_bias = np.column_stack((A1, np.ones(len(A1))))  
    Z2 = np.dot(H_bias, W2)
    
    # ✅ Z2 값 안정화 (클리핑)
    Z2 = np.clip(Z2, -10, 10)
    
    A2 = softmax(Z2)  
    return A2, Z2

# ✅ 시리얼 포트 설정
ser = serial.Serial("COM12", 115200)

# ✅ GSR 변화량 추적
prev_gsr = None

while True:
    try:
        # ✅ 시리얼 데이터 수신
        line = ser.readline().decode('utf-8').strip()
        values = line.split("\t") if "\t" in line else line.split()
        
        if len(values) < 4:
            print(f"⚠️ 잘못된 데이터 수신: {line}")
            continue

        # ✅ 심박수 & GSR 변환
        heart_rate = float(values[0])
        gsr = float(values[3])
        gsr_diff = gsr - prev_gsr if prev_gsr is not None else 0
        prev_gsr = gsr  

        # ✅ 입력 데이터 정규화 (학습 시 사용된 mean, scale로 변환)
        input_data = np.array([[heart_rate, gsr, gsr_diff]])
        input_data = (input_data - mean) / scale  # 정규화 적용

        # ✅ 감정 예측
        emotion_pred, Z2 = forward(input_data, W1, W2)

        # ✅ 확률 변환
        probabilities = np.round(emotion_pred[0] * 100, 2)

        # ✅ 감정 확률 출력
        prob_str = ", ".join([f"{emotion_mapping[label]}: {probabilities[label]}%" for label in range(len(probabilities))])
        print(f"📊 감정 확률 분포: {prob_str}")

        # ✅ Z2 값 확인 (디버깅)
        print(f"🔍 Z2 값 확인: {Z2[0]}")

        # ✅ 가장 높은 확률 감정 선택
        predicted_label = np.argmax(emotion_pred)
        predicted_emotion = emotion_mapping[predicted_label]
        print(f"🎯 실시간 감정 예측: {predicted_emotion} ({probabilities[predicted_label]}%)\n")

    except ValueError as ve:
        print(f"❌ 데이터 변환 오류: {ve} → 수신된 데이터: {line}")
    except Exception as e:
        print(f"❌ 기타 오류 발생: {e}")  
