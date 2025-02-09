import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# MySQL 연결
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0101",
    database="sensor_db"
)

cursor = db.cursor()

# MySQL에서 데이터 가져오기 
cursor.execute("SELECT timestamp, heart_rate, gsr FROM sensor_data ORDER BY timestamp DESC LIMIT 100")
data = cursor.fetchall()

# 데이터프레임 변환
df = pd.DataFrame(data, columns=["timestamp", "heart_rate", "gsr"])

# GSR 값 정규화
df["gsr_scaled"] = (df["gsr"] - df["gsr"].min()) / (df["gsr"].max() - df["gsr"].min()) * (df["heart_rate"].max() - df["heart_rate"].min()) + df["heart_rate"].min()

# GSR 데이터 부드럽게 만들기 (이동 평균 적용)
df["gsr_smoothed"] = df["gsr"].rolling(window=5).mean()

# 그래프 설정
fig, ax1 = plt.subplots(figsize=(12,6))

# 심박수 그래프 (왼쪽 Y축, 빨간색)
ax1.set_xlabel("Time")
ax1.set_ylabel("Heart Rate", color="red")
ax1.plot(df["timestamp"], df["heart_rate"], marker="o", linestyle="-", color="red", label="Heart Rate")
ax1.tick_params(axis="y", labelcolor="red")

# GSR 그래프 (오른쪽 Y축, 파란색)
ax2 = ax1.twinx()
ax2.set_ylabel("GSR", color="blue")
ax2.plot(df["timestamp"], df["gsr_smoothed"], marker="s", linestyle="--", color="blue", label="GSR (Smoothed)")
ax2.tick_params(axis="y", labelcolor="blue")

# X축 설정
plt.xticks(rotation=45)

# 제목 추가
plt.title("Heart Rate & GSR Trend")

# 그래프 출력
plt.show()

db.close()
