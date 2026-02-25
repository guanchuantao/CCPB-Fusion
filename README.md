# CCPB-Fusion

**CCPB-Fusion: A Multi-Model Collaborative Bounding Box Fusion Framework for Lung Nodule Detection in CT Images**

## 📖 Introduction

This repository contains the official implementation of the **CCPB-Fusion** framework. This project proposes a novel ensemble detection method that integrates the strengths of four state-of-the-art deep learning algorithms: **YOLOv5**, **SSD**, **Faster R-CNN**, and **Mask R-CNN**.

By employing an improved **Two-Stage Target IoU Clustering** algorithm and a **Multi-Layer Perceptron (MLP)** based fusion strategy, CCPB-Fusion effectively addresses the challenges of scale variation and complex morphology in pulmonary nodules. The framework significantly enhances detection sensitivity (Recall) while maintaining high precision, minimizing false positives in clinical diagnostic scenarios.

## 📂 Directory Structure

The project is structured into separate modules for each base model and the fusion logic:

```text
CCPB-Fusion/
├── all_fusion/          # Core fusion logic (Data processing, MLP training, Evaluation)
├── faster_r_cnn/        # Faster R-CNN implementation
├── mask_rcnn/           # Mask R-CNN implementation
├── ssd/                 # SSD implementation
├── yolov5-fc/           # YOLOv5 implementation
└── README.md
```

## ⚙️ Installation and Requirements

To ensure flexibility and avoid dependency conflicts, the configuration requirements are managed **separately for each algorithm**.

Please navigate to the specific algorithm directory you wish to run or train and install the corresponding dependencies found in the `requirements.txt` file.

**Example:**

```bash
# Install dependencies for Faster R-CNN
cd faster_r_cnn
pip install -r requirements.txt

# Install dependencies for YOLOv5
cd ../yolov5-fc
pip install -r requirements.txt
```

**System Recommendations:**
*   **Python:** 3.8+
*   **PyTorch:** 1.11.0+
*   **CUDA:** 11.3 (Recommended for GPU acceleration)

## ⬇️ Pre-trained Weights

This framework utilizes **Transfer Learning** for the backbone networks of Faster R-CNN, Mask R-CNN, and SSD.

*   **Source:** The pre-trained weights are sourced directly from the **Official PyTorch (Torchvision)** framework.
*   **Usage:** When initializing these models, the code is configured to automatically download or load the official ImageNet pre-trained weights provided by `torchvision.models`. You do not need to manually download third-party weights for these backbones.
*   **Note for YOLOv5:** The YOLOv5 model is trained from scratch to better adapt to the specific feature distribution of CT images and does not rely on external pre-trained weights.

## 🚀 Usage

1.  **Data Preparation:** Ensure your dataset (e.g., VOC format) is structured correctly.
2.  **Training Base Models:** Train the four individual models using their respective scripts.
3.  **Fusion Training:** Use the scripts in the `all_fusion/` directory to:
    *   Generate fusion features (clustering and predictions).
    *   Train the MLP fusion network.
    *   Evaluate the final CCPB-Fusion performance.
