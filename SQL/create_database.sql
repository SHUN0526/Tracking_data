-- ✅ 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS sensor_db;
USE sensor_db;

-- ✅ 센서 데이터 저장 테이블 생성
CREATE TABLE IF NOT EXISTS sensor_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    heart_rate INT,       
    spo2 INT,             
    temperature FLOAT,    
    gsr INT,              
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_data (
    id INT AUTO_INCREMENT PRIMARY KEY,                  -- 각 데이터를 구분하는 고유 ID (자동 증가)
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,      -- 데이터가 저장되는 시간 (자동 기록)
    heart_rate INT,                                      -- 심박수 값
    spo2 INT,                                            -- 산소포화도 값
    temperature FLOAT,                                   -- 온도 값
    gsr INT,                                             -- GSR (피부전도도) 값
    gsr_diff FLOAT,                                      -- GSR 변화량 (이전 값과의 차이)
    predicted_emotion VARCHAR(50),                      -- 예측된 감정 (기쁨, 긴장, 평온, 기타 등)
    probability FLOAT                                    -- 예측된 감정의 확률 (정확도)
);
