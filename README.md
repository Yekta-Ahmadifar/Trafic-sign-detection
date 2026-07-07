# Autonomous Traffic Sign Detection & Classification

An end-to-end computer vision and deep learning system built using TensorFlow and Keras to detect and classify traffic signs. Designed to simulate the visual processing unit of an autonomous vehicle, this project scales from a custom convolutional neural network (CNN) baseline to an optimized MobileNetV2 transfer learning model, successfully recognizing 52 distinct categories of road signals.

---

## 🚀 Features
* **Multi-Class Classification:** Robustly categorizes 52 unique traffic signs (speed limits, turn restrictions, hazards, etc.).
* **Data Augmentation Pipeline:** Synthesizes realistic driving environmental factors (subtle camera rotations, zooming, and shifting brightness levels) to expand model generalization.
* **Architecture Comparison:** Evaluates a scratch-built Custom CNN against an industry-grade pre-trained backbone (**MobileNetV2**).
* **Production Inference Pipeline:** Features a self-contained single-image pipeline that instantly processes new images, calculates localized confidence scores, and maps numerical predictions back to human-readable names.

---

## 📊 Dataset Structure
The project utilizes structured folders to isolate training and evaluation data, maintaining data integrity during validation steps:

```text
├── DATA/            # Training and Validation image data (split 80/20 automatically)
│   ├── 0/           # Subfolders corresponding to numerical Class IDs
│   ├── 1/
│   └── ...
├── TEST/            # Isolated evaluation dataset kept hidden during training
│   ├── 0/
│   └── ...
└── labels.csv       # Dictionary mapping ClassId integers to human-readable text strings
