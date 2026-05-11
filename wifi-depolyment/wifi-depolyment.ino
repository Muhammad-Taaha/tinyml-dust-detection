#include "esp_camera.h"
#include "tiny_cnn_model.h"
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <WiFi.h>
#include <WebSocketsServer.h>   // install: WebSockets by Markus Sattler
#include <ArduinoJson.h>        // install: ArduinoJson by Benoit Blanchon

// ==========================
// WiFi credentials
// ==========================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ==========================
// Configuration (unchanged from your code)
// ==========================
#define LED_PIN              4
#define CONFIDENCE_THRESHOLD 0.5f

const int input_width    = 128;
const int input_height   = 128;
const int input_channels = 3;

// Camera pins (AI-Thinker ESP32-CAM — unchanged)
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
// TFLite globals (unchanged from your code)
// ==========================
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model*      tfl_model   = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// ==========================
// WebSocket server on port 81
// ==========================
WebSocketsServer webSocket(81);
bool wsClientConnected = false;

// Last result cache — sent immediately when app connects
struct Result {
    String label;
    float  dusty_prob;
    float  clean_prob;
    unsigned long timestamp;
} lastResult;

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
            wsClientConnected = true;

            // Send the most recent result immediately so app isn't blank
            if (lastResult.timestamp > 0) {
                broadcastResult(lastResult.label,
                                lastResult.dusty_prob,
                                lastResult.clean_prob);
            }
            break;
        }

        case WStype_DISCONNECTED:
            Serial.printf("[WS] Client #%u disconnected\n", clientId);
            wsClientConnected = false;
            break;

        case WStype_TEXT:
            // App sends "ping" as a heartbeat — reply with pong
            if (String((char*)payload) == "ping") {
                webSocket.sendTXT(clientId, "{\"type\":\"pong\"}");
            }
            break;

        default:
            break;
    }
}

// ==========================
// Build and broadcast JSON result
// ==========================
void broadcastResult(const String& label,
                     float dusty_prob,
                     float clean_prob)
{
    // Keep a copy for new clients
    lastResult.label      = label;
    lastResult.dusty_prob = dusty_prob;
    lastResult.clean_prob = clean_prob;
    lastResult.timestamp  = millis();

    StaticJsonDocument<256> doc;
    doc["type"]        = "inference";
    doc["label"]       = label;
    doc["dusty_prob"]  = serialized(String(dusty_prob, 4));
    doc["clean_prob"]  = serialized(String(clean_prob, 4));
    doc["confidence"]  = serialized(String(
                            label == "dusty" ? dusty_prob : clean_prob, 4));
    doc["uptime_ms"]   = millis();
    doc["ip"]          = WiFi.localIP().toString();

    String json;
    serializeJson(doc, json);
    webSocket.broadcastTXT(json);

    Serial.println("[WS] Broadcast: " + json);
}

// ==========================
// WiFi connection
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
        Serial.print("ESP32 IP Address: ");
        Serial.println(WiFi.localIP());   // ← copy this into the app Settings
        Serial.println("Open app → Settings → enter this IP → Save & Connect");
    } else {
        Serial.println("\nWiFi FAILED — running offline (LED only)");
    }
}

// ==========================
// Camera init (unchanged from your code)
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
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = FRAMESIZE_QVGA;   // 320×240
    config.jpeg_quality = 12;
    config.fb_count     = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        while (1);
    }
    Serial.println("Camera OK");
}

// ==========================
// Preprocess (unchanged from your code)
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
// Setup
// ==========================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32-CAM Solar Panel Classifier + WebSocket");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    initCamera();
    connectWiFi();

    // Start WebSocket server
    webSocket.begin();
    webSocket.onEvent(onWebSocketEvent);
    Serial.println("WebSocket server started on port 81");

    // Load TFLite model (unchanged from your code)
    tfl_model = tflite::GetModel(tiny_cnn_esp32_int8_tflite);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        while (1);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        tfl_model, resolver,
        tensor_arena, kTensorArenaSize,
        error_reporter
    );
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed");
        while (1);
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);

    Serial.printf("Input:  %dx%d\n",
                  input->dims->data[1],
                  input->dims->data[2]);
    Serial.printf("Output: %d\n", output->dims->data[1]);
    Serial.println("Ready — waiting for camera frames...");
}

// ==========================
// Main loop
// ==========================
void loop() {
    webSocket.loop();   // ← must be called every loop iteration

    static unsigned long lastCapture = 0;
    if (millis() - lastCapture < 3000) return;   // inference every 3 seconds
    lastCapture = millis();

    // ── Capture ──────────────────────────────────────
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }

    // ── Preprocess ───────────────────────────────────
    uint8_t* rgb888 = (uint8_t*)malloc(input_width * input_height * 3);
    if (!rgb888) {
        Serial.println("malloc failed");
        esp_camera_fb_return(fb);
        return;
    }

    preprocess(fb->buf, fb->width, fb->height, rgb888);
    esp_camera_fb_return(fb);   // return frame buffer ASAP to free memory

    // ── Copy to input tensor ─────────────────────────
    memcpy(input->data.uint8, rgb888, input_width * input_height * 3);
    free(rgb888);

    // ── Inference ────────────────────────────────────
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed");
        return;
    }

    // ── Read output (same as your code) ──────────────
    float dusty_prob = output->data.f[0];
    float clean_prob = 1.0f - dusty_prob;
    String label     = (dusty_prob > CONFIDENCE_THRESHOLD) ? "dusty" : "clean";

    // ── LED (unchanged from your code) ───────────────
    digitalWrite(LED_PIN, label == "dusty" ? HIGH : LOW);

    // ── Serial (same as your code) ───────────────────
    Serial.printf("Clean: %.4f  Dusty: %.4f  → %s\n",
                  clean_prob, dusty_prob, label.c_str());

    // ── WebSocket broadcast (new) ─────────────────────
    broadcastResult(label, dusty_prob, clean_prob);
}
