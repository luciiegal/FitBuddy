#include <Wire.h>
#include <Adafruit_LSM6DSOX.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_Sensor.h>
#include <BluetoothSerial.h>
  
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run `make menuconfig` to enable it
#endif
BluetoothSerial SerialBT;
Adafruit_LSM6DSOX lsm6dsox;
Adafruit_LIS3MDL lis3mdl;
void setup() {
  Serial.begin(9600);
  SerialBT.begin("ESP32_GAB");
  Wire.begin();
  if (!lsm6dsox.begin_I2C()) {
    Serial.println("Erreur : LSM6DSOX non détecté !");
    SerialBT.println("LSM6DSOX non détecté !");
    while (1) delay(10);
  }
  Serial.println("LSM6DSOX détecté !");
  SerialBT.println("LSM6DSOX détecté !");
  if (!lis3mdl.begin_I2C()) {
    Serial.println("Erreur : LIS3MDL non détecté !");
    SerialBT.println("LIS3MDL non détecté !");
    while (1) delay(10);
  }
  Serial.println("LIS3MDL détecté !");
  SerialBT.println("LIS3MDL détecté !");
}
void loop() {
  sensors_event_t accel, gyro, temp, mag;
  lsm6dsox.getEvent(&accel, &gyro, &temp);
  lis3mdl.getEvent(&mag);
  String message = "📊 IMU Data\n";
  message += "🔹 Accel (m/s²) → X: " + String(accel.acceleration.x, 2) +
              " Y: " + String(accel.acceleration.y, 2) +
              " Z: " + String(accel.acceleration.z, 2) + "\n";
  message += "🔹 Gyro (rad/s) → X: " + String(gyro.gyro.x, 2) +
              " Y: " + String(gyro.gyro.y, 2) +
              " Z: " + String(gyro.gyro.z, 2) + "\n";
  message += "🔹 Magnet (uT) → X: " + String(mag.magnetic.x, 2) +
              " Y: " + String(mag.magnetic.y, 2) +
              " Z: " + String(mag.magnetic.z, 2) + "\n";
  message += "🌡️ Température: " + String(temp.temperature, 2) + " °C\n";
  Serial.println(message);
  SerialBT.println(message);
  delay(100);
}
