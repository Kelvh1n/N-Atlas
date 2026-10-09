/**
 * @file esp32_glove_left.ino
 * @brief Left Hand ASL Glove Firmware for ESP32.
 * Reads 5 flex sensors (ADC1) and MPU-9250 9-DOF IMU, then streams
 * high-rate telemetry packets to the Python host over Wi-Fi UDP (or Bluetooth).
 */

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "config.h"

#if !USE_WIFI_UDP
#include "BluetoothSerial.h"
BluetoothSerial SerialBT;
#endif

// UDP Client instance
WiFiUDP udp;

// Sensor variables
uint16_t flexRaw[5] = {0};
int16_t ax = 0, ay = 0, az = 0;
int16_t gx = 0, gy = 0, gz = 0;
int16_t mx = 0, my = 0, mz = 0;

unsigned long lastSampleTime = 0;
unsigned long packetCount = 0;

// ==========================================
// MPU-9250 Register Helpers
// ==========================================
#define MPU_PWR_MGMT_1   0x6B
#define MPU_ACCEL_XOUT_H 0x3B
#define MPU_GYRO_XOUT_H  0x43
#define MPU_INT_PIN_CFG  0x37
#define AK8963_ADDR      0x0C
#define AK8963_CNTL      0x0A
#define AK8963_HXL       0x03

void writeByte(uint8_t address, uint8_t subAddress, uint8_t data) {
    Wire.beginTransmission(address);
    Wire.write(subAddress);
    Wire.write(data);
    Wire.endTransmission();
}

bool initMPU9250() {
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000); // 400 kHz Fast I2C
    delay(20);

    // Wake up MPU-9250
    writeByte(MPU9250_I2C_ADDR, MPU_PWR_MGMT_1, 0x00);
    delay(10);

    // Enable I2C bypass multiplexer so ESP32 can talk directly to AK8963 Magnetometer
    writeByte(MPU9250_I2C_ADDR, MPU_INT_PIN_CFG, 0x02);
    delay(10);

    // Put AK8963 into Continuous measurement mode 2 (100 Hz, 16-bit)
    writeByte(AK8963_ADDR, AK8963_CNTL, 0x16);
    delay(10);

    Serial.println("[IMU] MPU-9250 & AK8963 initialized.");
    return true;
}

void readMPU9250() {
    // Read 14 bytes: Accel (6) + Temp (2) + Gyro (6)
    Wire.beginTransmission(MPU9250_I2C_ADDR);
    Wire.write(MPU_ACCEL_XOUT_H);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU9250_I2C_ADDR, (uint8_t)14);

    if (Wire.available() >= 14) {
        ax = (Wire.read() << 8) | Wire.read();
        ay = (Wire.read() << 8) | Wire.read();
        az = (Wire.read() << 8) | Wire.read();
        int16_t tempRaw = (Wire.read() << 8) | Wire.read(); (void)tempRaw;
        gx = (Wire.read() << 8) | Wire.read();
        gy = (Wire.read() << 8) | Wire.read();
        gz = (Wire.read() << 8) | Wire.read();
    }

    // Read Magnetometer (7 bytes: X, Y, Z + ST2 status register)
    Wire.beginTransmission(AK8963_ADDR);
    Wire.write(AK8963_HXL);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)AK8963_ADDR, (uint8_t)7);

    if (Wire.available() >= 7) {
        uint8_t xl = Wire.read(); uint8_t xh = Wire.read();
        uint8_t yl = Wire.read(); uint8_t yh = Wire.read();
        uint8_t zl = Wire.read(); uint8_t zh = Wire.read();
        uint8_t st2 = Wire.read(); (void)st2;

        mx = (int16_t)(xh << 8 | xl);
        my = (int16_t)(yh << 8 | yl);
        mz = (int16_t)(zh << 8 | zl);
    }
}

// ==========================================
// Flex Sensor ADC Oversampling
// ==========================================
uint16_t readFilteredADC(uint8_t pin) {
    uint32_t sum = 0;
    for (int i = 0; i < ADC_OVERSAMPLES; i++) {
        sum += analogRead(pin);
        delayMicroseconds(20);
    }
    return (uint16_t)(sum / ADC_OVERSAMPLES);
}

void readAllFlexSensors() {
    flexRaw[0] = readFilteredADC(PIN_FLEX_THUMB);
    flexRaw[1] = readFilteredADC(PIN_FLEX_INDEX);
    flexRaw[2] = readFilteredADC(PIN_FLEX_MIDDLE);
    flexRaw[3] = readFilteredADC(PIN_FLEX_RING);
    flexRaw[4] = readFilteredADC(PIN_FLEX_PINKY);
}

// ==========================================
// Wi-Fi Connection
// ==========================================
void setupWiFi() {
    Serial.printf("[WIFI] Connecting to SSID: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(300);
        digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WIFI] Connected! Assigned IP: %s\n", WiFi.localIP().toString().c_str());
        digitalWrite(PIN_STATUS_LED, HIGH);
        udp.begin(UDP_PORT);
    } else {
        Serial.println("\n[WIFI] Connection failed. Check SSID/Password or router.");
    }
}

// ==========================================
// Setup & Main Loop
// ==========================================
void setup() {
    Serial.begin(115200);
    pinMode(PIN_STATUS_LED, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);

    // Configure 12-bit ADC attenuation (0 - 3.3V range)
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db);

    Serial.printf("\n====================================\n");
    Serial.printf("  %s GLOVE - ASL TELEMETRY NODE\n", HAND_NAME);
    Serial.printf("====================================\n");

    // Initialize I2C IMU
    initMPU9250();

#if USE_WIFI_UDP
    setupWiFi();
#else
    SerialBT.begin(DEVICE_BT_NAME);
    Serial.printf("[BT] Bluetooth Serial Started as '%s'\n", DEVICE_BT_NAME);
#endif
}

// ==========================================
// Inbuilt LED Control & Incoming Command Handling
// ==========================================
void flashLED(int count, int delayMs) {
    for (int i = 0; i < count; i++) {
        digitalWrite(PIN_STATUS_LED, LOW);
        delay(delayMs);
        digitalWrite(PIN_STATUS_LED, HIGH);
        delay(delayMs);
    }
}

void checkIncomingUDP() {
#if USE_WIFI_UDP
    int packetSize = udp.parsePacket();
    if (packetSize > 0) {
        char packetBuffer[256];
        int len = udp.read(packetBuffer, 255);
        if (len > 0) packetBuffer[len] = 0;

        if (strstr(packetBuffer, "PING") != NULL) {
            // Respond with PONG packet containing IP and RSSI
            char reply[160];
            snprintf(reply, sizeof(reply),
                "{\"hand\":\"%s\",\"cmd\":\"PONG\",\"pong\":true,\"ip\":\"%s\",\"rssi\":%d}\n",
                HAND_NAME, WiFi.localIP().toString().c_str(), WiFi.RSSI());
            udp.beginPacket(udp.remoteIP(), udp.remotePort());
            udp.write((const uint8_t*)reply, strlen(reply));
            udp.endPacket();

            // Flash LED 2 times to visually confirm ping
            flashLED(2, 60);
        } else if (strstr(packetBuffer, "BLINK") != NULL) {
            // Flash LED 5 times to physically identify glove
            flashLED(5, 100);
        }
    }
#endif
}

void loop() {
    unsigned long now = millis();

    // Manage Inbuilt Status LED & Reconnect:
    // - WL_CONNECTED: Solid ON (HIGH)
    // - DISCONNECTED: Rapid blink (150ms interval)
#if USE_WIFI_UDP
    if (WiFi.status() == WL_CONNECTED) {
        digitalWrite(PIN_STATUS_LED, HIGH); // Solid ON indicates active connection
        checkIncomingUDP();
    } else {
        static unsigned long lastBlink = 0;
        if (now - lastBlink >= 150) {
            lastBlink = now;
            digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
        }
    }
#else
    if (SerialBT.hasClient()) {
        digitalWrite(PIN_STATUS_LED, HIGH);
    } else {
        static unsigned long lastBlink = 0;
        if (now - lastBlink >= 250) {
            lastBlink = now;
            digitalWrite(PIN_STATUS_LED, !digitalRead(PIN_STATUS_LED));
        }
    }
#endif

    if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = now;

        // 1. Acquire sensor readings
        readAllFlexSensors();
        readMPU9250();

        // 2. Format JSON Packet
        char buffer[320];
        snprintf(buffer, sizeof(buffer),
            "{\"hand\":\"%s\",\"id\":%d,\"ts\":%lu,\"seq\":%lu,"
            "\"flex\":[%u,%u,%u,%u,%u],"
            "\"accel\":[%d,%d,%d],"
            "\"gyro\":[%d,%d,%d],"
            "\"mag\":[%d,%d,%d]}\n",
            HAND_NAME, HAND_ID, now, packetCount++,
            flexRaw[0], flexRaw[1], flexRaw[2], flexRaw[3], flexRaw[4],
            ax, ay, az,
            gx, gy, gz,
            mx, my, mz
        );

        // 3. Transmit packet
#if USE_WIFI_UDP
        if (WiFi.status() == WL_CONNECTED) {
            udp.beginPacket(HOST_PC_IP, UDP_PORT);
            udp.write((const uint8_t*)buffer, strlen(buffer));
            udp.endPacket();
        }
#else
        SerialBT.print(buffer);
#endif

        // Also echo to USB Serial for local inspection
        if (packetCount % 25 == 0) {
            Serial.print(buffer);
        }
    }
}

