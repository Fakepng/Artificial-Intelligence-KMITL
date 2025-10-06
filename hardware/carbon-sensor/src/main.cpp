#include <Arduino.h>
#include <SensirionI2cScd4x.h>
#include <Wire.h>
#include <WiFi.h>
#include "time.h"
#include <ESP_Google_Sheet_Client.h>
#include <PubSubClient.h>

// ================== Wi-Fi & LED CONFIG ==================
#define WIFI_SSID     "Fakepng-LTE"
#define WIFI_PASSWORD "iotengineering"

#define LED_PIN 15                 // Blink this LED when Wi-Fi is connected
#define LED_ACTIVE_HIGH true       // Set to false if your LED is active-low

// Blink timing (ms)
const unsigned long WIFI_BLINK_INTERVAL = 500;

// Wi-Fi reconnect strategy
const unsigned long WIFI_CHECK_INTERVAL = 3000;    // check every 3s
const unsigned long WIFI_BEGIN_BACKOFF  = 15000;   // after 15s of failure, call WiFi.begin again

// ================== MQTT CONFIG ==================
#define MQTT_BROKER_HOST     ""     // TODO: replace with your broker
#define MQTT_BROKER_PORT     1883                    // Standard MQTT port
#define MQTT_USERNAME        ""                      // Optional, leave blank if not required
#define MQTT_PASSWORD        ""                      // Optional, leave blank if not required
#define MQTT_CLIENT_ID_BASE  ""
#define MQTT_TOPIC_BASE      ""       // Telemetry base topic

const unsigned long MQTT_RECONNECT_INTERVAL = 5000;  // retry every 5s

// ================== Google Service Account CONFIG ==================
// Google Project ID
#define PROJECT_ID ""
// Service Account's client email
#define CLIENT_EMAIL ""
// Service Account's private key
const char PRIVATE_KEY[] PROGMEM = R"";

// The ID of the spreadsheet where you'll publish the data
const char spreadsheetId[] = "";

// NTP server to request epoch time
const char* ntpServer = "pool.ntp.org";

// ================== SCD41 / Misc ==================
#define NO_ERROR 0
SensirionI2cScd4x sensor;

static char errorMessage[64];
static int16_t error;

unsigned long epochTime;
unsigned long lastTime = 0;
unsigned long timerDelay = 30000;

// Wi-Fi helpers
unsigned long lastWifiCheck = 0;
unsigned long firstDisconnectAt = 0;
bool wasConnected = false;

WiFiClient mqttNetClient;
PubSubClient mqttClient(mqttNetClient);

unsigned long lastMqttAttempt = 0;
String mqttClientId;

// LED helpers
unsigned long lastBlinkToggle = 0;
bool ledState = false;

// ====== Forward decls ======
void tokenStatusCallback(TokenInfo info);
unsigned long getTime();
void handleWifi();
void updateWifiLed();
void handleMqtt();
bool ensureMqttConnected();
void publishToMqtt(unsigned long timestamp, uint16_t co2Concentration, float temperature, float humidity);
void PrintUint64(uint64_t& value) {
  Serial.print("0x");
  Serial.print((uint32_t)(value >> 32), HEX);
  Serial.print((uint32_t)(value & 0xFFFFFFFF), HEX);
}

// Function that gets current epoch time
unsigned long getTime() {
  time_t now;
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return 0;
  time(&now);
  return now;
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(9600);
  // while (!Serial) { delay(50); }

  // LED setup
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LED_ACTIVE_HIGH ? LOW : HIGH); // start OFF

  Wire.begin();
  sensor.begin(Wire, SCD41_I2C_ADDR_62);

  uint64_t serialNumber = 0;
  delay(30);
  // Clean state
  error = sensor.wakeUp();
  if (error != NO_ERROR) { errorToString(error, errorMessage, sizeof errorMessage); Serial.printf("wakeUp(): %s\n", errorMessage); }
  error = sensor.stopPeriodicMeasurement();
  if (error != NO_ERROR) { errorToString(error, errorMessage, sizeof errorMessage); Serial.printf("stopPeriodicMeasurement(): %s\n", errorMessage); }
  error = sensor.reinit();
  if (error != NO_ERROR) { errorToString(error, errorMessage, sizeof errorMessage); Serial.printf("reinit(): %s\n", errorMessage); }

  // Read serial
  error = sensor.getSerialNumber(serialNumber);
  if (error != NO_ERROR) {
    errorToString(error, errorMessage, sizeof errorMessage);
    Serial.printf("getSerialNumber(): %s\n", errorMessage);
  } else {
    Serial.print("serial number: "); PrintUint64(serialNumber); Serial.println();
  }

  // Time
  configTime(0, 0, ntpServer);

  GSheet.printf("ESP Google Sheet Client v%s\n\n", ESP_GOOGLE_SHEET_CLIENT_VERSION);

  // Wi-Fi start
  WiFi.persistent(false);
  WiFi.setAutoReconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  mqttClient.setServer(MQTT_BROKER_HOST, MQTT_BROKER_PORT);
  mqttClient.setKeepAlive(30);

  Serial.print("Connecting to Wi-Fi");
  unsigned long startWait = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startWait < 20000) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected with IP: ");
    Serial.println(WiFi.localIP());
    wasConnected = true;
    uint64_t efuseMac = ESP.getEfuseMac();
    mqttClientId = String(MQTT_CLIENT_ID_BASE);
    mqttClientId += "-";
    mqttClientId += String((uint32_t)(efuseMac >> 32), HEX);
    mqttClientId += String((uint32_t)(efuseMac & 0xFFFFFFFF), HEX);
    ensureMqttConnected();
  } else {
    Serial.println("Initial Wi-Fi connect timed out, watchdog will retry.");
    wasConnected = false;
    firstDisconnectAt = millis();
  }

  // Google Sheets token setup
  GSheet.setTokenCallback(tokenStatusCallback);
  GSheet.setPrerefreshSeconds(10 * 60);
  GSheet.begin(CLIENT_EMAIL, PROJECT_ID, PRIVATE_KEY);

  // Start SCD41 periodic measurement
  error = sensor.startPeriodicMeasurement();
  if (error != NO_ERROR) {
    errorToString(error, errorMessage, sizeof errorMessage);
    Serial.printf("startPeriodicMeasurement(): %s\n", errorMessage);
  }
}

// -------------------- LOOP --------------------
void loop() {
  handleWifi();     // keep Wi-Fi alive
  updateWifiLed();  // blink LED only when connected
  handleMqtt();     // keep MQTT session alive

  bool ready = GSheet.ready();

  if (ready && WiFi.status() == WL_CONNECTED && millis() - lastTime > timerDelay) {
    lastTime = millis();

    bool dataReady = false;
    uint16_t co2Concentration = 0;
    float temperature = 0.0;
    float relativeHumidity = 0.0;

    FirebaseJson response;
    FirebaseJson valueRange;

    Serial.println("\nAppend spreadsheet values...");
    Serial.println("----------------------------");

    // Wait for data ready (short, non-blocking loop)
    error = sensor.getDataReadyStatus(dataReady);
    if (error != NO_ERROR) {
      errorToString(error, errorMessage, sizeof errorMessage);
      Serial.printf("getDataReadyStatus(): %s\n", errorMessage);
      return;
    }
    unsigned long waitStart = millis();
    while (!dataReady && millis() - waitStart < 2000) {  // wait up to 2s
      delay(100);
      error = sensor.getDataReadyStatus(dataReady);
      if (error != NO_ERROR) {
        errorToString(error, errorMessage, sizeof errorMessage);
        Serial.printf("getDataReadyStatus(): %s\n", errorMessage);
        return;
      }
    }

    error = sensor.readMeasurement(co2Concentration, temperature, relativeHumidity);
    if (error != NO_ERROR) {
      errorToString(error, errorMessage, sizeof errorMessage);
      Serial.printf("readMeasurement(): %s\n", errorMessage);
      return;
    }

    Serial.print("CO2 concentration [ppm]: "); Serial.println(co2Concentration);
    Serial.print("Temperature [°C]: ");       Serial.println(temperature);
    Serial.print("Relative Humidity [RH]: "); Serial.println(relativeHumidity);

    epochTime = getTime();

    publishToMqtt(epochTime, co2Concentration, temperature, relativeHumidity);

    valueRange.add("majorDimension", "COLUMNS");
    valueRange.set("values/[0]/[0]", epochTime);
    valueRange.set("values/[1]/[0]", co2Concentration);
    valueRange.set("values/[2]/[0]", temperature);
    valueRange.set("values/[3]/[0]", relativeHumidity);

    // Append to Google Sheet
    bool success = GSheet.values.append(&response, spreadsheetId, "Sheet1!A1", &valueRange);
    if (success) {
      response.toString(Serial, true);
      valueRange.clear();
    } else {
      Serial.println(GSheet.errorReason());
    }
    Serial.println();
    Serial.println(ESP.getFreeHeap());
  }
}

// -------------------- Wi-Fi helpers --------------------
void handleWifi() {
  unsigned long now = millis();

  // Update disconnect tracking
  if (WiFi.status() != WL_CONNECTED) {
    if (wasConnected) {
      // Just transitioned to disconnected
      firstDisconnectAt = now;
      wasConnected = false;
      Serial.println("[WiFi] Lost connection, attempting to recover...");
      lastMqttAttempt = 0;  // pause MQTT reconnects while offline
      if (mqttClient.connected()) {
        mqttClient.disconnect();
      }
    }
  } else {
    if (!wasConnected) {
      Serial.print("[WiFi] Reconnected. IP: ");
      Serial.println(WiFi.localIP());
      lastMqttAttempt = 0;  // allow immediate MQTT reconnect attempt
      ensureMqttConnected();
    }
    wasConnected = true;
  }

  // Periodic checks to recover Wi-Fi
  if (now - lastWifiCheck >= WIFI_CHECK_INTERVAL) {
    lastWifiCheck = now;

    if (WiFi.status() != WL_CONNECTED) {
      // Try quick reconnect first
      WiFi.reconnect();

      // If it has been a while, restart the begin sequence
      if (now - firstDisconnectAt >= WIFI_BEGIN_BACKOFF) {
        Serial.println("[WiFi] Backoff exceeded; calling WiFi.begin(...) again");
        WiFi.disconnect(true, true);
        delay(100);
        WiFi.mode(WIFI_STA);
        WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
        firstDisconnectAt = now;  // reset timer
      }
    }
  }
}

void updateWifiLed() {
  static bool lastShownConnected = false;

  bool connected = (WiFi.status() == WL_CONNECTED);

  if (connected) {
    // Blink while connected
    unsigned long now = millis();
    if (now - lastBlinkToggle >= WIFI_BLINK_INTERVAL) {
      lastBlinkToggle = now;
      ledState = !ledState;
      digitalWrite(LED_PIN, (LED_ACTIVE_HIGH ? (ledState ? HIGH : LOW) : (ledState ? LOW : HIGH)));
    }
  } else {
    // Ensure LED is OFF when disconnected
    if (lastShownConnected) {
      // transitioned to disconnected, force LED off
      digitalWrite(LED_PIN, LED_ACTIVE_HIGH ? LOW : HIGH);
      ledState = false;
    }
  }
  lastShownConnected = connected;
}

void handleMqtt() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  if (!mqttClient.connected()) {
    ensureMqttConnected();
  }

  if (mqttClient.connected()) {
    mqttClient.loop();
  }
}

bool ensureMqttConnected() {
  if (mqttClient.connected()) {
    return true;
  }

  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  unsigned long now = millis();
  if (now - lastMqttAttempt < MQTT_RECONNECT_INTERVAL) {
    return false;
  }
  lastMqttAttempt = now;

  if (mqttClientId.isEmpty()) {
    uint64_t efuseMac = ESP.getEfuseMac();
    mqttClientId = String(MQTT_CLIENT_ID_BASE);
    mqttClientId += "-";
    mqttClientId += String((uint32_t)(efuseMac >> 32), HEX);
    mqttClientId += String((uint32_t)(efuseMac & 0xFFFFFFFF), HEX);
  }

  Serial.print("[MQTT] Connecting as ");
  Serial.print(mqttClientId);
  Serial.print(" to ");
  Serial.print(MQTT_BROKER_HOST);
  Serial.print(":");
  Serial.println(MQTT_BROKER_PORT);

  bool connected;
  if (strlen(MQTT_USERNAME) > 0) {
    connected = mqttClient.connect(mqttClientId.c_str(), MQTT_USERNAME, MQTT_PASSWORD);
  } else {
    connected = mqttClient.connect(mqttClientId.c_str());
  }

  if (connected) {
    Serial.println("[MQTT] Connected");
  } else {
    Serial.print("[MQTT] Connect failed, rc=");
    Serial.println(mqttClient.state());
  }

  return connected;
}

void publishToMqtt(unsigned long timestamp, uint16_t co2Concentration, float temperature, float humidity) {
  if (!mqttClient.connected()) {
    if (!ensureMqttConnected()) {
      Serial.println("[MQTT] Skipping publish: client not connected");
      return;
    }
  }

  FirebaseJson json;
  json.set("timestamp", (double)timestamp);
  json.set("co2", co2Concentration);
  json.set("temperature", temperature);
  json.set("humidity", humidity);

  String payload;
  json.toString(payload, false);

  String topic = MQTT_TOPIC_BASE;
  topic += "/telemetry";

  if (mqttClient.publish(topic.c_str(), payload.c_str())) {
    Serial.println("[MQTT] Published telemetry");
  } else {
    Serial.println("[MQTT] Publish failed");
  }
}

// -------------------- Token callback --------------------
void tokenStatusCallback(TokenInfo info) {
  if (info.status == token_status_error) {
    GSheet.printf("Token info: type = %s, status = %s\n", GSheet.getTokenType(info).c_str(), GSheet.getTokenStatus(info).c_str());
    GSheet.printf("Token error: %s\n", GSheet.getTokenError(info).c_str());
  } else {
    GSheet.printf("Token info: type = %s, status = %s\n", GSheet.getTokenType(info).c_str(), GSheet.getTokenStatus(info).c_str());
  }
}
