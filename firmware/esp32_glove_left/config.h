#ifndef CONFIG_LEFT_H
#define CONFIG_LEFT_H

// ==========================================
// Hand Identity
// ==========================================
#define HAND_NAME      "LEFT"
#define HAND_ID        0          // 0 = Left, 1 = Right
#define DEVICE_BT_NAME "ASL_Glove_LEFT"

// ==========================================
// Network Configuration (Wi-Fi & UDP)
// ==========================================
// Replace with your local Wi-Fi router or laptop Mobile Hotspot credentials
#define WIFI_SSID       "ASL_GLOVE_NET"
#define WIFI_PASS       "12345678"

// Host PC IP address where the Python GUI is running
// Can also use UDP broadcast e.g., "192.168.4.255" or your laptop IP
#define HOST_PC_IP      "192.168.4.1"
#define UDP_PORT        5005

// Streaming mode: Set to 1 for Wi-Fi UDP (recommended), 0 for Bluetooth Serial
#define USE_WIFI_UDP    1

// ==========================================
// Pin Definitions (Strictly ADC1 for Wi-Fi)
// ==========================================
#define PIN_FLEX_THUMB   36   // ADC1_CH0 (VP)
#define PIN_FLEX_INDEX   39   // ADC1_CH3 (VN)
#define PIN_FLEX_MIDDLE  34   // ADC1_CH6
#define PIN_FLEX_RING    35   // ADC1_CH7
#define PIN_FLEX_PINKY   32   // ADC1_CH4

#define PIN_I2C_SDA      21
#define PIN_I2C_SCL      22

#define PIN_STATUS_LED    2   // Onboard Blue LED

// ==========================================
// Sensor & Sampling Parameters
// ==========================================
#define SAMPLE_INTERVAL_MS 20    // 50 Hz streaming rate
#define ADC_OVERSAMPLES    8     // Oversampling filter to reject noise
#define MPU9250_I2C_ADDR   0x68

#endif // CONFIG_LEFT_H
