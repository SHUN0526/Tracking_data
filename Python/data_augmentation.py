import pandas as pd
import numpy as np

# ✅ 기존 데이터 불러오기
df = pd.read_csv("processed_sensor_data.csv")

# ✅ -1 데이터 필터링 (심박수 또는 GSR이 -1인 행 제거)
df = df[(df["heart_rate"] != -1) & (df["gsr"] != -1)]

# ✅ 각 감정별 데이터 개수 확인
label_counts = df["emotion_label"].value_counts()
max_count = label_counts.max()  # 가장 많은 감정 개수를 기준으로 증강
print(f"📊 감정별 데이터 개수:\n{label_counts}\n")

# ✅ 랜덤 노이즈 추가 함수
def add_noise(value, noise_level=0.05):
    return value + np.random.uniform(-noise_level, noise_level) * value

# ✅ 증강할 데이터 저장 리스트
augmented_data = []

# ✅ 각 감정별로 균등한 데이터 증강
for label, count in label_counts.items():
    if count < max_count:  # 가장 많은 개수에 맞추기 위해 부족한 데이터만 증강
        required_count = max_count - count  # 증강해야 하는 개수
        augmentation_factor = required_count // count  # 증강 배수
        remainder = required_count % count  # 남은 개수 처리

        # ✅ 해당 감정 데이터만 선택
        subset_df = df[df["emotion_label"] == label]

        # ✅ 증강 수행
        for _ in range(augmentation_factor):
            for _, row in subset_df.iterrows():
                new_row = row.copy()
                new_row["heart_rate"] = add_noise(row["heart_rate"], noise_level=0.05)
                new_row["gsr"] = add_noise(row["gsr"], noise_level=0.1)
                new_row["gsr_diff"] = add_noise(row["gsr_diff"], noise_level=0.1)
                augmented_data.append(new_row)

        # ✅ 남은 개수 추가 증강 (정확한 균형을 위해)
        extra_rows = subset_df.sample(n=remainder, replace=True)
        for _, row in extra_rows.iterrows():
            new_row = row.copy()
            new_row["heart_rate"] = add_noise(row["heart_rate"], noise_level=0.05)
            new_row["gsr"] = add_noise(row["gsr"], noise_level=0.1)
            new_row["gsr_diff"] = add_noise(row["gsr_diff"], noise_level=0.1)
            augmented_data.append(new_row)

# ✅ 증강된 데이터프레임 생성
augmented_df = pd.DataFrame(augmented_data)

# ✅ 기존 데이터와 합치기
final_df = pd.concat([df, augmented_df], ignore_index=True)

# ✅ 증강된 데이터 저장
final_df.to_csv("augmented_sensor_data.csv", index=False)

# ✅ 최종 감정별 데이터 개수 확인
final_counts = final_df["emotion_label"].value_counts()
print(f"✅ 데이터 증강 완료! 기존 {len(df)}개 → {len(final_df)}개")
print(f"📊 최종 감정별 데이터 개수:\n{final_counts}")
