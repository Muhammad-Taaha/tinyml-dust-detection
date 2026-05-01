# Solar Panel Dust Detection for ESP32-CAM – TinyML Pipeline

This repository documents **multiple approaches** to classify solar panels as **clean** or **dusty** using a cropped panel dataset. The final solution is a **tiny (<65 KB) quantized CNN** that runs on an ESP32‑CAM **without external PSRAM**, achieving ~82‑85% validation accuracy. All scripts, trained models, and deployment code are included.

---

## 📦 Project Structure
```
cleaned-tiny-ml/
│
├── data/                              # Dataset (not uploaded to Git due to size)
│   └── cropped_dataset/
│       ├── train/
│       │   ├── clean/                 # Clean panel images for training
│       │   └── dusty/                 # Dusty panel images for training
│       └── val/
│           ├── clean/
│           └── dusty/
│
├── models/                            # Trained model files (ignored by Git)
│   ├── mobilenetv2_transfer.keras     # MobileNetV2 transfer learning (12 MB)
│   ├── tiny_cnn_esp32.keras           # Final tiny CNN (581 KB)
│   └── tiny_cnn_esp32_int8.tflite     # INT8 quantized for ESP32 (65 KB)
│
├── scripts/                           # Python training & conversion scripts
│   ├── train_tiny_cnn.py              # Train the ESP32‑optimized CNN
│   ├── train_mobilenet.py             # Fine‑tune MobileNetV2
│   ├── quantize_model.py              # Convert to INT8 TFLite
│   ├── balance_checker.py             # Check dataset class distribution
│   └── finetune_solar.py              # MCUNetV2 fine‑tune experiment
│
├── esp32/                             # ESP32‑CAM firmware
│   ├── dust_detector.ino              # Main inference + MQTT
│   └── wifi_deployment.ino            # Wi‑Fi configuration helper
│
├── mobile-app/                        # React Native (Expo) mobile app
│   └── DustDetector/                  # Full Expo project
│       ├── App.js                     # Main app component
│       ├── hooks/                     # Custom hooks (WebSocket, notif.)
│       ├── screens/                   # UI screens (Dashboard, History, Settings)
│       └── package.json
│
├── docs/                              # Documentation & assets
│   └── readme.md                      (old readme – can be merged)
│
├── .gitignore                         # Ignores data/, models/, node_modules, etc.
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview (this file)
```
---

## 📊 Dataset Preparation

The original dataset consists of 160×160 images with YOLO‑format bounding boxes (classes: `clean`, `dusty`). To remove background variation, we **crop** each image to the bounding box, then resize to 128×128 (colour). The resulting cropped dataset:
cropped_dataset/
train/clean/ → 856 images
train/dusty/ → 879 images
val/clean/ → 269 images
val/dusty/ → 234 images


Class distribution is balanced (ratio ≈ 1:1).  
Run `python balance_checker.py` to verify.

---

## 🧠 Explored Approaches

### 1. YOLOv5n (Object Detection)
- Trained on original 160×160 images with bounding boxes.
- Achieved **85% mAP50** on test set.
- **Problem**: Model size ~5 MB, needs >4 MB PSRAM and runs at 1‑2 FPS on ESP32‑CAM.  
  Not suitable for standard ESP32 without PSRAM.

### 2. MCUNetV2 (MIT HAN Lab)
- Attempted to fine‑tune `mcunet-in0` classification model on cropped panels.
- Encountered **broken download links** (404), missing pre‑trained weights, and outdated toolchain.
- After manually downloading weights, training from scratch gave low accuracy (~66%).  
  Abandoned due to high complexity and unreliable infrastructure.

### 3. MobileNetV2 Transfer Learning (best accuracy)
- Used pretrained MobileNetV2 (ImageNet) as feature extractor, input size 224×224.
- Two‑phase training: first freeze base, train head; then unfreeze top 30 layers for fine‑tuning.
- Achieved **87.9% validation accuracy**.
- **Problem**: Model size ~12 MB, after INT8 quantisation ~3‑4 MB – still requires PSRAM.  
  Retained as reference for higher accuracy systems.

### 4. Custom Tiny CNN (final solution)
- Designed a **depthwise‑separable CNN** with squeeze‑excitation blocks and residual connections.
- Input: 128×128 colour, mild augmentation (horizontal flip, rotation, zoom, brightness).
- Training from scratch with L2 regularisation, dropout 0.4, label smoothing.
- Achieved **~82‑85% validation accuracy**.
- **Advantages**: Model size after INT8 quantisation **65 KB**, RAM usage <200 KB.  
  Runs smoothly on standard ESP32‑CAM **without PSRAM**.

---

## 🚀 How to Train the Tiny CNN

1. **Prepare cropped dataset** as described above.
2. Install dependencies:
   ```bash
   pip install tensorflow numpy pillow
