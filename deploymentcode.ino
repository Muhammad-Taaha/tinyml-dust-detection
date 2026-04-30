#include "esp_camera.h"
#include "tiny_cnn_model.h"   // your converted model header
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ==========================
// Configuration
// ==========================
#define LED_PIN 4            // Built‑in LED on AI‑Thinker ESP32‑CAM (GPIO4)
#define CONFIDENCE_THRESHOLD 0.5   // adjust as needed

const int input_width = 128;
const int input_height = 128;
const int input_channels = 3;

// Camera pins for AI‑Thinker ESP32‑CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// TensorFlow Lite globals
static tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Arena size (adjust if needed – 64KB should be enough for this tiny model)
constexpr int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// ==========================
// Function prototypes
// ==========================
void initCamera();
void preprocess(uint8_t* image_data, int width, int height, uint8_t* out_buffer);

// ==========================
// Setup
// ==========================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 Solar Panel Clean/Dusty Classifier");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  initCamera();

  // Load the quantized TFLite model
  model = tflite::GetModel(tiny_cnn_esp32_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  // Register all operations
  static tflite::AllOpsResolver resolver;
  interpreter = new tflite::MicroInterpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed");
    while (1);
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print model info
  Serial.print("Input size: ");
  Serial.print(input->dims->data[1]);
  Serial.print("x");
  Serial.println(input->dims->data[2]);
  Serial.print("Output channels: ");
  Serial.println(output->dims->data[1]);   // 1 for binary classification
}

// ==========================
// Main loop
// ==========================
void loop() {
  // Capture image
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    return;
  }

  // The camera returns JPEG by default. We need to convert to RGB.
  // For simplicity, we will set camera to RGB565 (grayscale) or use JPEG decoding.
  // Here we assume we have the camera configured for RGB565 (see initCamera).
  // The 'fb->buf' contains RGB565 data. We'll convert to RGB888 and resize.

  uint8_t* rgb888 = (uint8_t*)malloc(input_width * input_height * 3);
  if (!rgb888) {
    Serial.println("Memory allocation failed");
    esp_camera_fb_return(fb);
    return;
  }

  // Resize and convert to RGB888 (128x128)
  // Use a simple nearest‑neighbor resize for speed (adapt if needed)
  // We'll implement a basic resizing function.
  // Note: This function assumes the camera frame is in RGB565 format.
  // If you use JPEG, you must decode it first (see notes below).
  preprocess(fb->buf, fb->width, fb->height, rgb888);

  // Copy the preprocessed image into the input tensor (uint8)
  memcpy(input->data.uint8, rgb888, input_width * input_height * 3);

  free(rgb888);
  esp_camera_fb_return(fb);

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Inference failed");
    return;
  }

  // Output is a single float (probability of dusty)
  // The model was exported with int8 output, but the interpreter returns float
  // if we didn't force int8 output. Actually, the quantized model outputs int8,
  // but TFLite Micro will dequantize it to float automatically.
  float dusty_prob = output->data.f[0];   // between 0 and 1
  float clean_prob = 1.0 - dusty_prob;

  Serial.print("Clean: ");
  Serial.print(clean_prob, 4);
  Serial.print(", Dusty: ");
  Serial.println(dusty_prob, 4);

  if (dusty_prob > CONFIDENCE_THRESHOLD) {
    digitalWrite(LED_PIN, HIGH);
    Serial.println("⚠️ DUSTY panel detected");
  } else {
    digitalWrite(LED_PIN, LOW);
    Serial.println("✓ Clean panel");
  }

  delay(2000);   // wait 2 seconds before next capture
}

// ==========================
// Camera initialization
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
  config.pixel_format = PIXFORMAT_RGB565;   // we need RGB data
  config.frame_size = FRAMESIZE_QVGA;        // 320x240 (enough for resizing)
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    while (1);
  }
  Serial.println("Camera OK");
}

// ==========================
// Resize RGB565 to 128x128 RGB888 (nearest neighbor)
// ==========================
void preprocess(uint8_t* rgb565_data, int src_width, int src_height, uint8_t* out_rgb888) {
  float x_ratio = (float)src_width / input_width;
  float y_ratio = (float)src_height / input_height;

  for (int y = 0; y < input_height; y++) {
    int src_y = (int)(y * y_ratio);
    if (src_y >= src_height) src_y = src_height - 1;
    for (int x = 0; x < input_width; x++) {
      int src_x = (int)(x * x_ratio);
      if (src_x >= src_width) src_x = src_width - 1;

      int idx = (src_y * src_width + src_x);
      uint16_t rgb565 = ((uint16_t*)rgb565_data)[idx];

      // Convert RGB565 to RGB888
      uint8_t r = (rgb565 >> 11) & 0x1F;
      uint8_t g = (rgb565 >> 5) & 0x3F;
      uint8_t b = rgb565 & 0x1F;
      r = (r << 3) | (r >> 2);
      g = (g << 2) | (g >> 4);
      b = (b << 3) | (b >> 2);

      int out_idx = (y * input_width + x) * 3;
      out_rgb888[out_idx] = r;
      out_rgb888[out_idx + 1] = g;
      out_rgb888[out_idx + 2] = b;
    }
  }
}