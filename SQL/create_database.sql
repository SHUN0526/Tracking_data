-- 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS sensor_db;
USE sensor_db;

-- 센서 데이터 저장 테이블 생성
CREATE TABLE IF NOT EXISTS sensor_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    heart_rate INT,       
    spo2 INT,             
    temperature FLOAT,    
    gsr INT,              
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
