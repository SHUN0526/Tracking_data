#include <bluefruit.h>
#include <DFRobot_BloodOxygen_S.h>

#define I2C_ADDRESS 0x57
#define GSR_PIN A0

DFRobot_BloodOxygen_S_I2C MAX30102(&Wire, I2C_ADDRESS);

// ✅ BLE 서비스 및 캐릭터리스틱
BLEService emotionService = BLEService("0000FFF0-0000-1000-8000-00805F9B34FB");
BLECharacteristic sensorCharacteristic = BLECharacteristic("0000FFF1-0000-1000-8000-00805F9B34FB");

unsigned long lastReadTime = 0;
const int readInterval = 1000;  // 1초 간격

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // ✅ 센서 초기화
  Wire.begin();
  Wire.setClock(100000);

  Serial.println("센서 초기화 중...");
  int retryCount = 0;
  while (retryCount < 50 && !MAX30102.begin()) {
    Serial.println("MAX30102 실패, 재시도 중...");
    retryCount++;
    delay(1000);
  }
  if (retryCount == 50) {
    Serial.println("센서 초기화 실패. 프로그램 중단.");
    while (1);
  }
  MAX30102.sensorStartCollect();

  // ✅ BLE 초기화
  Bluefruit.begin();
  Bluefruit.setName("MyDeviceName");

  emotionService.begin();
  sensorCharacteristic.setProperties(CHR_PROPS_NOTIFY);
  sensorCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  sensorCharacteristic.setFixedLen(4);
  sensorCharacteristic.begin();
  Bluefruit.Advertising.addService(emotionService);

  Bluefruit.Advertising.addName();
  Bluefruit.Advertising.start();
  Serial.println("🔵 BLE 광고 시작됨: MyDeviceName");
}

void loop() {
  unsigned long currentTime = millis();
  if (currentTime - lastReadTime >= readInterval) {
    lastReadTime = currentTime;

    // GSR 평균
    long gsrSum = 0;
    for (int i = 0; i < 10; i++) {
      gsrSum += analogRead(GSR_PIN);
      delay(5);
    }
    int gsrAverage = gsrSum / 10;

    // MAX30102 데이터
    MAX30102.getHeartbeatSPO2();
    int spo2Value = MAX30102._sHeartbeatSPO2.SPO2;
    int heartRateValue = MAX30102._sHeartbeatSPO2.Heartbeat;
    float temperature = MAX30102.getTemperature_C();

    Serial.print("HR:"); Serial.print(heartRateValue);
    Serial.print(" SPO2:"); Serial.print(spo2Value);
    Serial.print(" TEMP:"); Serial.print(temperature);
    Serial.print(" GSR:"); Serial.println(gsrAverage);

    // BLE 전송
    uint8_t data[4] = {
      (uint8_t)heartRateValue,
      (uint8_t)spo2Value,
      (uint8_t)gsrAverage,
      0  // reserved
    };
    sensorCharacteristic.notify(data, sizeof(data));
  }
}