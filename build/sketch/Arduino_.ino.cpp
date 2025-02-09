#include <Arduino.h>
#line 1 "c:\\arduino_jol\\Sensor_Code\\Arduino_.ino"
#include <DFRobot_BloodOxygen_S.h>

#define I2C_ADDRESS 0x57
#define GSR_PIN A0  // GSR 센서가 연결된 핀

DFRobot_BloodOxygen_S_I2C MAX30102(&Wire, I2C_ADDRESS);

unsigned long lastReadTime = 0;
const int readInterval = 1000;  // 1초마다 데이터 읽기

#line 11 "c:\\arduino_jol\\Sensor_Code\\Arduino_.ino"
void setup();
#line 35 "c:\\arduino_jol\\Sensor_Code\\Arduino_.ino"
void loop();
#line 11 "c:\\arduino_jol\\Sensor_Code\\Arduino_.ino"
void setup() {
  Serial.begin(115200);
  while (!Serial);
  Wire.begin();
  Wire.setClock(400000);

  Serial.println("MAX30102 및 GSR 초기화 시작!");

  int retryCount = 0;
  while (retryCount < 50 && !MAX30102.begin()) {
    Serial.println("MAX30102 초기화 실패! 다시 시도 중...");
    retryCount++;
    delay(1000);
  }

  if (retryCount == 50) {
    Serial.println("MAX30102 초기화 실패: 프로그램 중단");
    while (1);
  }

  Serial.println("MAX30102 초기화 성공!");
  MAX30102.sensorStartCollect();
}

void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastReadTime >= readInterval) {
    lastReadTime = currentTime;

    long gsrSum = 0;
    for (int i = 0; i < 10; i++) {
      gsrSum += analogRead(GSR_PIN);
      delay(5);
    }
    int gsrAverage = gsrSum / 10;

    MAX30102.getHeartbeatSPO2();

    int spo2Value = MAX30102._sHeartbeatSPO2.SPO2;
    int heartRateValue = MAX30102._sHeartbeatSPO2.Heartbeat;
    float temperature = MAX30102.getTemperature_C();

    Serial.print(heartRateValue);
    Serial.print("\t");
    Serial.print(spo2Value);
    Serial.print("\t");
    Serial.print(temperature, 2);
    Serial.print("\t");
    Serial.println(gsrAverage);
  }
}

