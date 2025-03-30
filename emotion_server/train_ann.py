import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from itertools import product

# ✅ 사용자 JSON 파일 로드
with open("emotion_labeling_input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ JSON → DataFrame 변환
sensor_records = []
for item in data["emotion_labeling_data"]:
    emotion = item["label"]
    for record in item["labeled_data"]:
        record["emotion"] = emotion
        sensor_records.append(record)

df = pd.DataFrame(sensor_records)

# ✅ 전처리: 결측값 및 -1 제거
df.replace(-1, np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ GSR 변화량 계산
df["gsr_diff"] = df["gsr"].diff().fillna(0)

# ✅ 감정 라벨 인코딩
label_encoder = LabelEncoder()
df["emotion_label"] = label_encoder.fit_transform(df["emotion"])
emotion_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print("📊 감정 라벨 매핑:", emotion_mapping)

# ✅ 학습 입력값/정답값 설정
X = df[["heart_rate", "gsr", "gsr_diff"]].values
y = df["emotion_label"].values

# ✅ 입력 정규화
mean = X.mean(axis=0)
scale = X.std(axis=0)
X = (X - mean) / scale

# ✅ 원-핫 인코딩
num_classes = len(np.unique(y))
y_one_hot = np.zeros((len(y), num_classes))
y_one_hot[np.arange(len(y)), y] = 1

# ✅ 소프트맥스 함수 (안정성 개선)
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    #제일 큰 값 빼기
    exp_x = np.exp(x)
    #시그모이드 함수 적용
    return exp_x / exp_x.sum(axis=1, keepdims=True)
    #모든 합이 1로 되도록 함

# ✅ 시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ✅ 순전파 (Forward Propagation)
def forward(X, W1, W2):
    X_bias = np.column_stack((X, np.ones(len(X))))#(1,3+1)
    Z1 = np.dot(X_bias, W1)#(1,3+1),(3+1,h) -- (1,h)
    A1 = sigmoid(Z1)#(1,h)  

    H_bias = np.column_stack((A1, np.ones(len(A1))))#(1,h+1)  
    Z2 = np.dot(H_bias, W2)#(1,h+1),(h+1,4)
    A2 = softmax(Z2)#(1,4)  
    return A2, Z1, A1, Z2, X_bias, H_bias

# ✅ 역전파 (Backpropagation)
def backward(X, X_bias, H_bias, y, A2, Z1, A1, W1, W2, lr):
    out_err = A2 - y#(1,4)-y
    out_delta = out_err  

    hid_err = out_delta.dot(W2[:-1, :].T)#(1,4)*(4,h) -> (1,h)
    hid_delta = hid_err * (A1 * (1 - A1))#(1,h) x곱x ((1,h) x 1-(1,h))

    #경사하강법
    W2 -= H_bias.T.dot(out_delta) * lr  #(h+1,1),(1,4) -> (h+1,4)
    W1 -= X_bias.T.dot(hid_delta) * lr  #(3+1,1),(1,h) -> (3+1,h)

    return W1, W2

# ✅ 하이퍼파라미터 최적화
hidden_layer_sizes = range(4, 10)
learning_rates = [0.01, 0.005, 0.001]
epochs = 5000  

best_accuracy = 0
best_params = {}

print("\n🔎 하이퍼파라미터 탐색 시작...\n")
results = []

for hidden_size, learning_rate in product(hidden_layer_sizes, learning_rates):
    print(f"\n🚀 테스트 중 - 은닉층: {hidden_size}, 학습률: {learning_rate}")

    # ✅ 모델 초기화
    input_size = X.shape[1]
    #.shape[1]-열의 개수 (3개)
    output_size = num_classes
    #0,1,2,3 - 4개
  
    #초기 가중치값
    W1 = np.random.randn(input_size + 1, hidden_size) * 0.01
    W2 = np.random.randn(hidden_size + 1, output_size) * 0.01

    mse_list = []
    acc_list = []

    for epoch in range(epochs):
        A2, Z1, A1, Z2, X_bias, H_bias = forward(X, W1, W2)
        W1, W2 = backward(X, X_bias, H_bias, y_one_hot, A2, Z1, A1, W1, W2, learning_rate)

        mse = np.mean((y_one_hot - A2) ** 2)
        mse_list.append(mse)

        predictions = np.argmax(A2, axis=1)#(1,4)
        accuracy = np.mean(predictions == y)
        acc_list.append(accuracy)

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"📢 Epoch {epoch}/{epochs} - MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")

    results.append({
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "accuracy": accuracy
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "accuracy": accuracy
        }
        # ✅ 모델 저장 (정규화 정보도 함께 저장)
        with open("best_ann_model.pkl", "wb") as f:
            pickle.dump((W1, W2, emotion_mapping, mean, scale), f)

print("\n🎯 최적의 하이퍼파라미터 탐색 완료!\n")
print(f"✅ 은닉층 개수: {best_params['hidden_size']}")
print(f"✅ 학습률: {best_params['learning_rate']}")
print(f"✅ 에포크: {best_params['epochs']}")
print(f"✅ 최종 정확도: {best_params['accuracy']:.4f}")
print("📦 모델 저장 위치: best_ann_model.pkl")
print("📊 라벨 매핑:", emotion_mapping)


# ✅ 최적의 하이퍼파라미터 저장
with open("best_hyperparameters.pkl", "wb") as f:
    pickle.dump(best_params, f)
