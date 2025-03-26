import mysql.connector
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ✅ MySQL 데이터 가져오기
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0101",
    database="sensor_db"
)
cursor = db.cursor()
cursor.execute("SELECT timestamp, heart_rate, gsr FROM sensor_data ORDER BY timestamp DESC LIMIT 600")
data = cursor.fetchall()
db.close()

# ✅ 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(data, columns=["timestamp", "heart_rate", "gsr"])

# ✅ -1 값이 포함된 데이터 제거
df = df[(df["heart_rate"] != -1) & (df["gsr"] != -1)]

# ✅ GSR 변화량(ΔGSR) 계산
df["gsr_diff"] = df["gsr"].diff().fillna(0)

# ✅ 감정 상태 분류 기준 설정
heart_mean, heart_std = df["heart_rate"].mean(), df["heart_rate"].std()
gsr_mean, gsr_std = df["gsr"].mean(), df["gsr"].std()
gsr_diff_std = df["gsr_diff"].std()

# ✅ 감정 상태 분류 함수
def classify_emotion(row):
    hr, gsr, gsr_diff = row["heart_rate"], row["gsr"], row["gsr_diff"]

    if hr > heart_mean + heart_std and abs(gsr_diff) < gsr_diff_std and abs(gsr - gsr_mean) < gsr_std:
        return "기쁨"
    elif hr > heart_mean + heart_std and gsr > gsr_mean + (gsr_std*0.3):
        return "긴장"
    elif (
        abs(hr - heart_mean) < heart_std
        and abs(gsr - gsr_mean) < gsr_std
    ):
        return "평온"
    else:
        return "기타"

df["emotion"] = df.apply(classify_emotion, axis=1)

# ✅ 감정 라벨 인코딩 수행
label_encoder = LabelEncoder()
df["emotion_label"] = label_encoder.fit_transform(df["emotion"])

# ✅ 감정 라벨 매핑 확인
emotion_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("📢 감정 라벨 매핑:\n", emotion_mapping)

# ✅ 데이터 저장
df.to_csv("processed_sensor_data.csv", index=False)
print(f"✅ 데이터 전처리 완료! {len(df)}개 데이터 저장됨.")
