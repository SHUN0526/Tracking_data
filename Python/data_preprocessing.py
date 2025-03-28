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
cursor = db.cursor()#데이터베이스 커서
cursor.execute("SELECT timestamp, heart_rate, gsr FROM sensor_data ORDER BY timestamp DESC LIMIT 600")
#ASC는 반대
data = cursor.fetchall()
#execute한 결과를 가져옴-리스트의 각 요소는 튜플(튜플은 수정 불가)로 구성
db.close()

# ✅ 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(data, columns=["timestamp", "heart_rate", "gsr"])
#pd.DataFrame()은 행과 열로 구성된 테이블 형식-ann 돌리기 위함
#data의 timestamp, heart_rate, gsr 이용
# ✅ -1 값이 포함된 데이터 제거
df = df[(df["heart_rate"] != -1) & (df["gsr"] != -1)]

# ✅ GSR 변화량(ΔGSR) 계산
df["gsr_diff"] = df["gsr"].diff().fillna(0)
#데이터프레임에 새로운 열(gsr_diff) 추가, fillna(0)은 첫 행을 0으로 함-NaN을 삭제

# ✅ 감정 상태 분류 기준 설정
heart_mean, heart_std = df["heart_rate"].mean(), df["heart_rate"].std()
gsr_mean, gsr_std = df["gsr"].mean(), df["gsr"].std()
gsr_diff_std = df["gsr_diff"].std()

# ✅ 감정 상태 분류 함수
def classify_emotion(row):
#한 행을 입력으로 받음-(row)
    hr, gsr, gsr_diff = row["heart_rate"], row["gsr"], row["gsr_diff"]
    #한 행씩 처리하기 때문에 row이름 사용
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
#행 단위로 입력함-axis=1

# ✅ 감정 라벨 인코딩 수행-단어를 숫자로 변환-0,1,2,3
label_encoder = LabelEncoder()
df["emotion_label"] = label_encoder.fit_transform(df["emotion"])

# ✅ 감정 라벨 매핑 확인
#zip은 두 리스트를 쌍으로 묶어줌-[('기쁨',0),('긴장',1),...]
emotion_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("📢 감정 라벨 매핑:\n", emotion_mapping)

# ✅ 데이터 저장-timestamp, heart_rate, gsr, gsr_diff, emotion, emotion_label
df.to_csv("processed_sensor_data.csv", index=False, encoding="utf-8-sig")
print(f"✅ 데이터 전처리 완료! {len(df)}개 데이터 저장됨.")
