import numpy as np
import pandas as pd
import pickle
from itertools import product

# ✅ 시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ✅ 순전파 (Forward Propagation)
def forward(X, W1, W2):
    X_bias = np.column_stack((X, np.ones(len(X))))
    Z1 = np.dot(X_bias, W1)
    A1 = sigmoid(Z1)

    H_bias = np.column_stack((A1, np.ones(len(A1))))
    Z2 = np.dot(H_bias, W2)
    A2 = sigmoid(Z2)
    return A2, Z1, A1, Z2, X_bias, H_bias

# ✅ 역전파 (Backpropagation)
def backward(X, X_bias, H_bias, y, A2, Z1, A1, W1, W2, lr):
    out_err = y - A2
    out_delta = out_err * sigmoid_derivative(A2)

    hid_err = out_delta.dot(W2[:-1, :].T)
    hid_delta = hid_err * sigmoid_derivative(A1)

    W2 += H_bias.T.dot(out_delta) * lr
    W1 += X_bias.T.dot(hid_delta) * lr

    return W1, W2

# ✅ 데이터 로드
df = pd.read_csv("augmented_sensor_data.csv")
df = df[(df["heart_rate"] != -1) & (df["gsr"] != -1)]

# ✅ 감정 라벨 개수 확인 (기타(Other) 제외)
emotion_counts = df[df["emotion_label"] != 1]["emotion_label"].value_counts().sort_index()
print("📢 감정 라벨 개수 확인 (기타 제외):\n", emotion_counts)

# ✅ 가장 적은 개수를 가진 감정 찾기 & 가중치 설정
min_label = emotion_counts.idxmin()  # 가장 적은 개수를 가진 감정 라벨
weight_label = min_label  # 해당 감정에 가중치 적용
weight_value = 20  # 가중치 1.5배 적용

print(f"✅ 가장 적은 개수를 가진 감정: {weight_label}, 가중치 {weight_value} 적용됨.")

# ✅ 입력값 (X)와 정답값 (y)
X = df[["heart_rate", "gsr", "gsr_diff"]].values
y = df["emotion_label"].values.reshape(-1, 1)

# ✅ 데이터 정규화 (평균 0, 표준편차 1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# ✅ 실험할 하이퍼파라미터 범위 설정 (에포크는 10,000으로 고정)
hidden_layer_sizes = range(1, 15)  # 은닉층 뉴런 개수 (1 ~ 15)
learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]  # 학습률 범위
epochs = 10000  # 에포크는 10,000으로 고정

# ✅ 최적의 하이퍼파라미터 저장 변수
best_accuracy = 0
best_params = {}

# ✅ 모든 조합을 시도하여 최적값 찾기
for hidden_size, learning_rate in product(hidden_layer_sizes, learning_rates):
    print(f"\n🚀 테스트 중 - 은닉층: {hidden_size}, 학습률: {learning_rate}, 에포크: {epochs}")

    # ✅ 모델 초기화
    input_size = X.shape[1]
    output_size = 1
    W1 = np.random.randn(input_size + 1, hidden_size) * 0.01
    W2 = np.random.randn(hidden_size + 1, output_size) * 0.01

    # ✅ 학습 실행
    for epoch in range(epochs):
        A2, Z1, A1, Z2, X_bias, H_bias = forward(X, W1, W2)
        W1, W2 = backward(X, X_bias, H_bias, y, A2, Z1, A1, W1, W2, learning_rate)

        # ✅ MSE (Mean Squared Error) 계산
        mse = np.mean((y - A2) ** 2)

        # ✅ Accuracy (정확도) 계산
        predictions = np.round(A2)
        accuracy = np.mean(predictions == y)

        if epoch % 2000 == 0:
            print(f"📢 Epoch {epoch}/{epochs} - MSE: {mse:.4f}, Accuracy: {accuracy:.4f}")

    # ✅ 최적의 하이퍼파라미터 업데이트 (가장 높은 정확도를 기준으로)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "accuracy": accuracy
        }
        # ✅ 최적의 모델 저장
        with open("best_ann_model.pkl", "wb") as f:
            pickle.dump((W1, W2), f)

print("\n🎯 최적의 하이퍼파라미터 찾음!")
print(f"✅ 은닉층: {best_params['hidden_size']}, 학습률: {best_params['learning_rate']}, 에포크: {best_params['epochs']}")
print(f"✅ 최종 Accuracy: {best_params['accuracy']:.4f}")

# ✅ 최적의 하이퍼파라미터 저장
with open("best_hyperparameters.pkl", "wb") as f:
    pickle.dump(best_params, f)
