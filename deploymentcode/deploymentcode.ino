#include "esp_camera.h"
#include "tiny_cnn_model.h"

#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic pop

// ==========================
// WiFi Credentials
// ==========================
const char* WIFI_SSID     = "NetComm 8165";
const char* WIFI_PASSWORD = "Cusenagatu";

// ==========================
// Servers
// ==========================
WebServer server(80);
WebSocketsServer webSocket(81);

// ==========================
// HTML Dashboard
// ==========================
const char* htmlPage = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 Dust Monitor</title>
  <style>
    body { font-family: Arial; text-align: center; background: #111; color: #FFD700; }
    h1 { margin-top: 20px; }
    .box { font-size: 24px; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>ESP32 Dust Monitor</h1>
  <div class="box">Status: <span id="label">--</span></div>
  <div class="box">Confidence: <span id="conf">--</span></div>

  <script>
    const ws = new WebSocket("ws://" + location.hostname + ":81");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "inference") {
        document.getElementById("label").innerText = data.label;
        document.getElementById("conf").innerText =
          (data.confidence * 100).toFixed(2) + "%";
      }
    };
  </script>
</body>
</html>
)rawliteral";

// ==========================
// Camera + ML config
// ==========================
#define LED_PIN 4
#define CONFIDENCE_THRESHOLD 0.5f

const int input_width = 128;
const int input_height = 128;
const int input_channels = 3;

// Camera pins (AI Thinker)
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// ==========================
// TensorFlow Lite globals
// ==========================
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;
static tflite::AllOpsResolver resolver;

constexpr int TENSOR_ARENA_SIZE = 300 * 1024;
static uint8_t* tensor_arena = nullptr;

// ==========================
// Result storage
// ==========================
struct Result {
  String label;
  float dusty;
  float clean;
  unsigned long ts;
} lastResult;

// ==========================
// WebSocket broadcast
// ==========================
void broadcastResult(String label, float dusty, float clean) {
  lastResult = {label, dusty, clean, millis()};

  StaticJsonDocument<256> doc;
  doc["type"] = "inference";
  doc["label"] = label;
  doc["dusty_prob"] = dusty;
  doc["clean_prob"] = clean;
  doc["confidence"] = (label == "dusty") ? dusty : clean;

  String json;
  serializeJson(doc, json);

  webSocket.broadcastTXT(json);
  Serial.println(json);
}

// ==========================
// WebSocket events
// ==========================
void onWebSocket(uint8_t num, WStype_t type, uint8_t *payload, size_t len) {
  if (type == WStype_CONNECTED) {
    Serial.println("Client connected");
  }
}

// ==========================
// Camera init
// ==========================
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QVGA;
  config.fb_count = 1;

  esp_camera_init(&config);
}

// ==========================
// WiFi
// ==========================
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());
}

// ==========================
// Setup
// ==========================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  tensor_arena = (uint8_t*) ps_malloc(TENSOR_ARENA_SIZE);

  initCamera();
  connectWiFi();

  // ================= HTTP SERVER =================
  server.on("/", []() {
    server.send(200, "text/html", htmlPage);
  });
  server.begin();
  Serial.println("HTTP server started");

  // ================= WEBSOCKET =================
  webSocket.begin();
  webSocket.onEvent(onWebSocket);

  // ================= MODEL =================
  model = tflite::GetModel(tiny_cnn_esp32_int8_tflite);

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);

  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

// ==========================
// Loop
// ==========================
void loop() {
  webSocket.loop();
  server.handleClient();

  static unsigned long last = 0;
  if (millis() - last < 3000) return;
  last = millis();

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  // Fake inference (keep your ML code here unchanged)
  float dusty = random(0, 100) / 100.0;
  float clean = 1 - dusty;
  String label = (dusty > CONFIDENCE_THRESHOLD) ? "dusty" : "clean";

  digitalWrite(LED_PIN, label == "dusty");

  broadcastResult(label, dusty, clean);

  esp_camera_fb_return(fb);
}