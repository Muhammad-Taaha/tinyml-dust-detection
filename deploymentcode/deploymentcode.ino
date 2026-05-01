#include "esp_camera.h"
#include "tiny_cnn_model.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#pragma GCC diagnostic 

#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

// ==========================
// ⚠️ CHANGE THESE
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// ==========================

#define LED_PIN              4
#define CONFIDENCE_THRESHOLD 0.5f

const int input_width    = 128;
const int input_height   = 128;
const int input_channels = 3;

// ESP32-CAM AI-Thinker pins
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
// TFLite globals
// ==========================
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::ErrorReporter*     error_reporter = &micro_error_reporter;
static const tflite::Model*       tfl_model      = nullptr;
static tflite::MicroInterpreter*  interpreter    = nullptr;
static TfLiteTensor*              tfl_input      = nullptr;
static TfLiteTensor*              tfl_output     = nullptr;
static tflite::AllOpsResolver     resolver;

// ✅ PSRAM allocated arena
constexpr int kTensorArenaSize = 300 * 1024;
static uint8_t* tensor_arena   = nullptr;

// ==========================
// WebSocket server on port 81
// ==========================
WebSocketsServer webSocket(81);

struct Result {
  String label;
  float  dusty_prob;
  float  clean_prob;
  unsigned long timestamp;
} lastResult;

// ==========================
// Forward declarations
// ==========================
void broadcastResult(const String& label, float dusty_prob, float clean_prob);

// ==========================
// WebSocket event handler
// ==========================
void onWebSocketEvent(uint8_t clientId,
                      WStype_t type,
                      uint8_t* payload,
                      size_t length)
{
  switch (type) {
    case WStype_CONNECTED: {
      IPAddress ip = webSocket.remoteIP(clientId);
      Serial.printf("[WS] Client #%u connected from %s\n",
                    clientId, ip.toString().c_str());
      if (lastResult.timestamp > 0) {
        broadcastResult(lastResult.label,
                        lastResult.dusty_prob,
                        lastResult.clean_prob);
      }
      break;
    }
    case WStype_DISCONNECTED:
      Serial.printf("[WS] Client #%u disconnected\n", clientId);
      break;
    case WStype_TEXT:
      if (String((char*)payload) == "ping") {
        webSocket.sendTXT(clientId, "{\"type\":\"pong\"}");
      }
      break;
    default:
      break;
  }
}

// ==========================
// Broadcast JSON over WebSocket
// ==========================
void broadcastResult(const String& label, float dusty_prob, float clean_prob)
{
  lastResult.label      = label;
  lastResult.dusty_prob = dusty_prob;
  lastResult.clean_prob = clean_prob;
  lastResult.timestamp  = millis();

  StaticJsonDocument<256> doc;
  doc["type"]       = "inference";
  doc["label"]      = label;
  doc["dusty_prob"] = dusty_prob;
  doc["clean_prob"] = clean_prob;
  doc["confidence"] = (label == "dusty") ? dusty_prob : clean_prob;
  doc["uptime_ms"]  = millis();
  doc["ip"]         = WiFi.localIP().toString();

  String json;
  serializeJson(doc, json);
  webSocket.broadcastTXT(json);
  Serial.println("[WS] Sent: " + json);
}

// ==========================
// WiFi
// ==========================
void connectWiFi() {
  Serial.printf("Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
    Serial.println("Use this IP in your app settings.");
  } else {
    Serial.println("\nWiFi FAILED — LED only mode");
  }
}

// ==========================
// Camera init
// ==========================
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;  // needed for preprocessing
  config.frame_size   = FRAMESIZE_QVGA;   // 320x240
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  if (psramFound()) {
    config.fb_count = 2;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed: 0x%x\n", err);
    while (1);
  }
  Serial.println("Camera OK");
}

// ==========================
// Preprocess: RGB565 320x240 → RGB888 128x128
// ==========================
void preprocess(uint8_t* rgb565_data,
                int src_width, int src_height,
                uint8_t* out_rgb888)
{
  float x_ratio = (float)src_width  / input_width;
  float y_ratio = (float)src_height / input_height;

  for (int y = 0; y < input_height; y++) {
    int src_y = (int)(y * y_ratio);
    if (src_y >= src_height) src_y = src_height - 1;

    for (int x = 0; x < input_width; x++) {
      int src_x = (int)(x * x_ratio);
      if (src_x >= src_width) src_x = src_width - 1;

      uint16_t rgb565 = ((uint16_t*)rgb565_data)[src_y * src_width + src_x];

      uint8_t r = (rgb565 >> 11) & 0x1F;
      uint8_t g = (rgb565 >> 5)  & 0x3F;
      uint8_t b =  rgb565        & 0x1F;

      // Expand to 8-bit
      r = (r << 3) | (r >> 2);
      g = (g << 2) | (g >> 4);
      b = (b << 3) | (b >> 2);

      int out_idx = (y * input_width + x) * 3;
      out_rgb888[out_idx]     = r;
      out_rgb888[out_idx + 1] = g;
      out_rgb888[out_idx + 2] = b;
    }
  }
}

// ==========================
// Copy RGB888 into input tensor (handles all tensor types)
// ==========================
void fillInputTensor(uint8_t* rgb888) {
  int total = input_width * input_height * input_channels;

  if (tfl_input->type == kTfLiteUInt8) {
    memcpy(tfl_input->data.uint8, rgb888, total);

  } else if (tfl_input->type == kTfLiteInt8) {
    for (int i = 0; i < total; i++) {
      tfl_input->data.int8[i] = (int8_t)((int)rgb888[i] - 128);
    }
  } else {
    // float32
    for (int i = 0; i < total; i++) {
      tfl_input->data.f[i] = rgb888[i] / 255.0f;
    }
  }
}

// ==========================
// Read output tensor → dusty probability
// ==========================
float readOutput() {
  if (tfl_output->type == kTfLiteUInt8) {
    return (tfl_output->data.uint8[0] - tfl_output->params.zero_point)
           * tfl_output->params.scale;

  } else if (tfl_output->type == kTfLiteInt8) {
    return (tfl_output->data.int8[0] - tfl_output->params.zero_point)
           * tfl_output->params.scale;
  }
  // float32
  return tfl_output->data.f[0];
}

// ==========================
// Setup
// ==========================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\nESP32-CAM Dust Detector");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // Allocate tensor arena in PSRAM
  tensor_arena = (uint8_t*) ps_malloc(kTensorArenaSize);
  if (!tensor_arena) {
    Serial.println("[ERROR] PSRAM allocation failed!");
    while (1);
  }
  Serial.printf("Tensor arena: %d KB in PSRAM\n", kTensorArenaSize / 1024);

  initCamera();
  connectWiFi();

  // Start WebSocket server
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  Serial.println("WebSocket server started on port 81");

  // Load model
  tfl_model = tflite::GetModel(tiny_cnn_esp32_int8_tflite);
  if (tfl_model == nullptr) {
    Serial.println("[ERROR] Failed to load model!");
    while (1);
  }

  static tflite::MicroInterpreter static_interpreter(
      tfl_model, resolver,
      tensor_arena, kTensorArenaSize,
      error_reporter
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors failed!");
    Serial.printf("Free heap:  %u\n", ESP.getFreeHeap());
    Serial.printf("Free PSRAM: %u\n", ESP.getFreePsram());
    while (1);
  }

  tfl_input  = interpreter->input(0);
  tfl_output = interpreter->output(0);

  Serial.printf("Input  dims: %dx%dx%d  type: %d\n",
                tfl_input->dims->data[1],
                tfl_input->dims->data[2],
                tfl_input->dims->data[3],
                tfl_input->type);
  Serial.printf("Output dims: %d  type: %d\n",
                tfl_output->dims->data[1],
                tfl_output->type);
  Serial.printf("Arena used:  %u bytes\n", interpreter->arena_used_bytes());
  Serial.println("Model ready! Inference every 3s.");
}

// ==========================
// Loop
// ==========================
void loop() {
  webSocket.loop();  // must be called every iteration

  static unsigned long lastCapture = 0;
  if (millis() - lastCapture < 3000) return;
  lastCapture = millis();

  // Capture frame
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ERROR] Camera capture failed");
    return;
  }

  // Allocate RGB888 buffer
  uint8_t* rgb888 = (uint8_t*) malloc(input_width * input_height * input_channels);
  if (!rgb888) {
    Serial.println("[ERROR] malloc failed for rgb888");
    esp_camera_fb_return(fb);
    return;
  }

  // Preprocess: resize + convert RGB565 → RGB888
  preprocess(fb->buf, fb->width, fb->height, rgb888);
  esp_camera_fb_return(fb);  // return ASAP to free DMA memory

  // Fill input tensor
  fillInputTensor(rgb888);
  free(rgb888);

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Invoke failed!");
    return;
  }

  // Read result
  float dusty_prob = readOutput();
  float clean_prob = 1.0f - dusty_prob;
  String label     = (dusty_prob > CONFIDENCE_THRESHOLD) ? "dusty" : "clean";

  // LED
  digitalWrite(LED_PIN, label == "dusty" ? HIGH : LOW);

  // Serial
  Serial.printf("Clean: %.4f  Dusty: %.4f  → %s\n",
                clean_prob, dusty_prob, label.c_str());

  // WebSocket broadcast
  broadcastResult(label, dusty_prob, clean_prob);
}