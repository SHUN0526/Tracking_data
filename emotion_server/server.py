from flask import Flask, request, jsonify
import json
import subprocess
import os

app = Flask(__name__)

@app.route("/train_model", methods=["POST"])
def train_model():
    try:
        data = request.get_json()
        print("📥 감정 라벨링 데이터 수신:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # 받은 데이터를 저장
        with open("emotion_labeling_input.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # ✅ 경로를 명확히 지정
        script_path = os.path.join(os.path.dirname(__file__), "train_ann.py")
        subprocess.run(["python", script_path], check=True)

        return jsonify("SUCCESS!")
    except Exception as e:
        print("❌ 오류 발생:", e)
        return jsonify(f"ERROR: {str(e)}"), 500


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    try:
        input_data = request.get_json()

        # 센서값 추출
        heart_rate = input_data.get("heart_rate")
        spo2 = input_data.get("spo2", None)
        temperature = input_data.get("temperature")
        gsr = input_data.get("gsr")
        gsr_diff = input_data.get("gsr_diff", 0)

        # 모델 로드
        import pickle
        import numpy as np

        with open("best_ann_model.pkl", "rb") as f:
            W1, W2, emotion_mapping, mean, scale = pickle.load(f)

        # 정규화
        X = np.array([[heart_rate, gsr, gsr_diff]])
        X = (X - mean) / scale

        # 순전파 함수
        def sigmoid(x): return 1 / (1 + np.exp(-x))
        # ✅ 개선된 Softmax 함수
        def softmax(x, scale_factor=0.1):#한쪽 값만 크게 예측하지 않도록 *0.1함
            x = x * scale_factor
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        def forward(X, W1, W2):
            X_bias = np.column_stack((X, np.ones(len(X))))
            Z1 = np.dot(X_bias, W1)
            A1 = sigmoid(Z1)
            H_bias = np.column_stack((A1, np.ones(len(A1))))
            Z2 = np.dot(H_bias, W2)
            A2 = softmax(Z2)
            return A2

        # 예측
        predictions = forward(X, W1, W2)
        predicted_label = int(np.argmax(predictions))

        # emotion_mapping이 dict 형태면 문자열 키인지, int 키인지 확인
        if isinstance(emotion_mapping, dict):
            # 라벨이 숫자형 키인지 문자열 키인지 여부에 따라
            # 예: emotion_mapping["0"] or emotion_mapping[0]
            if str(predicted_label) in emotion_mapping:
                predicted_emotion = emotion_mapping[str(predicted_label)]
            else:
                predicted_emotion = emotion_mapping.get(predicted_label, "Unknown")
        else:
            predicted_emotion = emotion_mapping[predicted_label]

        probabilities = predictions[0].tolist()

        return jsonify({
            "predicted_emotion": predicted_emotion,
            "probabilities": probabilities
        })

    except Exception as e:
        print("❌ 예측 오류:", e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/backup_emotion_data", methods=["POST"])
def backup_emotion_data():
    try:
        data = request.get_json()
        print("📦 백업 데이터 수신:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        with open("emotion_data_backup.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return jsonify("Store SUCCESS!")
    except Exception as e:
        print("❌ 백업 오류:", e)
        return jsonify(f"ERROR: {str(e)}"), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
