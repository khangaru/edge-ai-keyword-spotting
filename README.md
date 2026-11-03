# Edge AI Wake Word Detection System (`"Hey KAISTian"`)

An end-to-end embedded machine learning pipeline designed to detect a custom wake word on resource-constrained edge hardware (Raspberry Pi). This project covers data collection, audio Digital Signal Processing (MFCC), neural network architecture design, Post-Training Quantization (PTG) to INT8, hardware benchmarking, and real-time streaming optimization.

---

## 🚀 Key Features
* **Custom Audio Pipeline:** Curated audio samples with Voice Activity Detection (VAD), DC offset removal, amplification, downsampling, and Mel-Frequency Cepstral Coefficient (MFCC) feature extraction.
* **Data Augmentation:** Expanded small datasets from 240 to 1,200 training samples using waveform augmentation to prevent overfitting and improve acoustic generalization.
* **Model Quantization:** Converted Float32 models to INT8 TensorFlow Lite (`.tflite`) format, achieving a **~39% file size reduction** (from 22.6 KB down to 13.8 KB) with minimal accuracy penalty.
* **Hardware Benchmarking:** Deployed and evaluated models on a Raspberry Pi, tracking inference latency, memory footprints, and error metrics (RMSE, MAE, False Positive/Negative rates).
* **Streaming Inference Engine:** Implemented sliding-window classification, confidence threshold tuning, consecutive-frame temporal smoothing, and CSV event logging.

---

## 📂 Repository Structure
```text
├── models/             # Trained Float32 and INT8 TFLite models
├── notebooks/          # Google Colab training & MelSpectrogram extraction scripts
├── reports/            # Full technical project report (PDF)
├── src/                # Streaming inference, VAD, and hardware benchmarking code
└── README.md
