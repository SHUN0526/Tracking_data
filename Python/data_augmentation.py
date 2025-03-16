import pandas as pd
import numpy as np

# ✅ 기존 데이터 불러오기
df = pd.read_csv("processed_sensor_data.csv")

# ✅ -1 데이터 필터링 (심박수 또는 GSR이 -1인 행 제거)
df = df[(df["heart_rate"] != -1) & (df["gsr"] != -1)]

# ✅ 데이터 증강을 위한 빈 리스트
augmented_data = []

# ✅ 증강할 데이터 개수 (기존 데이터의 3배 추가)
augmentation_factor = 0

# ✅ 랜덤 노이즈 추가 함수
def add_noise(value, noise_level=0.05):
    return value + np.random.uniform(-noise_level, noise_level) * value

# ✅ 데이터 증강 수행
for _ in range(augmentation_factor):
    for _, row in df.iterrows():
        new_row = row.copy()
        
        # ✅ 심박수 변형 (랜덤 노이즈 추가)
        new_row["heart_rate"] = add_noise(row["heart_rate"], noise_level=0.05)

        # ✅ GSR 변형 (랜덤 노이즈 추가)
        new_row["gsr"] = add_noise(row["gsr"], noise_level=0.1)

        # ✅ GSR 변화량 변형 (노이즈 추가)
        new_row["gsr_diff"] = add_noise(row["gsr_diff"], noise_level=0.1)

        # ✅ 감정 라벨 유지 (원본과 동일한 감정)
        new_row["emotion"] = row["emotion"]
        new_row["emotion_label"] = row["emotion_label"]

        # ✅ 새로운 데이터 추가
        augmented_data.append(new_row)

# ✅ 증강된 데이터프레임 생성
augmented_df = pd.DataFrame(augmented_data)

# ✅ 기존 데이터와 합치기
final_df = pd.concat([df, augmented_df], ignore_index=True)

# ✅ 증강된 데이터 저장
final_df.to_csv("augmented_sensor_data.csv", index=False)

print(f"✅ 데이터 증강 완료! 기존 {len(df)}개 → {len(final_df)}개")
