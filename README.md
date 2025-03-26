# Deep Learning Models for Retina Image Classification

This study provides a comprehensive comparison of different deep learning models for **retina image classification**. Five models—**CNN, CNN+Attention, CNN+Attention+LSTM, Swin-Transformer, and ViT (Vision Transformer)**—were evaluated based on accuracy, memory usage, training time, and energy consumption.

## 1. Project Overview

Retinal images play a crucial role in medical diagnostics, particularly for the **early detection of diseases such as glaucoma, diabetic retinopathy, and cataracts**. This study utilizes **deep learning models** to classify different eye conditions.

### The models analyzed in this study include:
- **CNN (Convolutional Neural Network)**
- **CNN + Attention (CNN Enhanced with Attention Mechanism)**
- **CNN + Attention + LSTM (CNN Supported with Long Short-Term Memory Layer)**
- **Swin-Transformer (Swin-based Transformer for Image Classification)**
- **ViT (Vision Transformer for Image Classification)**

### Key Evaluation Metrics:
- **Accuracy**
- **Precision, Recall, and F1 Score**
- **GPU and RAM Usage**
- **Training Time**
- **Energy Consumption**

### Key Findings:
- **Swin Transformer achieved the highest accuracy at 92%.**
- **ViT model followed with 88% accuracy.**
- **CNN + LSTM + Attention provided 86% accuracy and stood out as a sustainable low-resource alternative.**
- **While Transformer-based models achieved superior accuracy, they required significantly higher GPU and energy consumption.**

## 2. Usage Instructions

### 2.1. Requirements
Ensure that the following Python libraries are installed before running the scripts:

```bash
pip install torch torchvision timm scikit-learn matplotlib numpy opencv-python
```

These libraries support model training, testing, data preprocessing, and visualization.

### 2.2. Training and Testing the Models
Each deep learning model is implemented as a separate `.py` script. Use the commands below to run a specific model:

- **To train and test the CNN model:**
  ```bash
  python CNN.py
  ```

- **To train and test the CNN+Attention model:**
  ```bash
  python CNN+Attention.py
  ```

- **To train and test the CNN+Attention+LSTM model:**
  ```bash
  python CNN+Attention+LSTM.py
  ```

- **To train and test Swin-Transformer and ViT models:**
  ```bash
  python swin_vit.py
  ```

This script includes both **Swin-Transformer** and **Vision Transformer (ViT)** models. The user can specify which model to run within the script.

## 3. Dataset

This study uses a **retinal image dataset** publicly available on **Kaggle**. It provides essential medical images for deep learning-based **disease classification**.

🔗 **Kaggle Dataset:** [Medical Scan Classification Dataset](https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset?select=Retinal+Imaging)

### Dataset Classes:
- **Normal Eye (Healthy Retina)**
- **Glaucoma (Optic Nerve Damage)**
- **Diabetic Retinopathy (Diabetes-Related Eye Disease)**
- **Cataract (Clouding of the Eye Lens)**

### 3.1. Image Preprocessing Steps
To improve data quality and model efficiency, the following preprocessing steps were applied:

- **Contrast Enhancement:** Histogram equalization was used to enhance image contrast.
- **Resizing:**
  - CNN-based models: **256x256 pixels**
  - Transformer-based models: **512x512 pixels**
- **Tensor Conversion:** Images were converted into tensor format for compatibility with PyTorch models.
- **Normalization:** Pixel values were normalized to stabilize training and improve convergence.

## 4. Model Performance Comparison

The table below presents the comparative performance results for the five models tested:

| Model | Accuracy (%) | Precision | Recall | F1 Score |
|--------|-------------|------------|--------|---------|
| CNN | 58.0 | 57.2 | 56.8 | 57.0 |
| CNN + Attention | 84.0 | 83.5 | 83.1 | 83.3 |
| CNN + Attention + LSTM | 86.0 | 85.7 | 85.3 | 85.5 |
| ViT | 88.0 | 87.6 | 87.3 | 87.5 |
| Swin Transformer | 92.0 | 91.8 | 91.4 | 91.6 |

Results indicate that **Swin Transformer achieved the highest accuracy**, while **CNN + LSTM + Attention emerged as a viable alternative with lower energy consumption**.

## 5. Energy Consumption and Resource Utilization

| Model | GPU RAM Usage (MB) | Energy Consumption (W) | Training Time (s) |
|--------|----------------------|-----------------|-----------------|
| CNN + LSTM + Attention | 6612 | 31.22 | 734.58 |
| ViT | 12194.4 | 77.63 | 4617.81 |
| Swin Transformer | 23200 | 73.74 | 4756.75 |

The **CNN + LSTM + Attention model demonstrated lower power consumption and hardware requirements**, making it an energy-efficient alternative. Meanwhile, Transformer-based models required **significantly higher GPU memory and power consumption**, though they achieved superior classification performance.

## 6. Future Research Directions

Based on the findings of this study, several areas for improvement have been identified:

- **Enhancing Model Efficiency:** Optimizing hyperparameters to improve classification accuracy.
- **Using Larger Datasets:** Expanding the dataset to assess the generalization capabilities of the models.
- **Comparative Analysis with Other AI Models:** Including ResNet, EfficientNet, and DenseNet for further benchmarking.

## 7. References

For academic citations, please refer to this study as follows:

```
Saygılı, A. & Ömer, A.   (2025). Enhancing Retinal Disease Detection with the Swin Transformer: A Comprehensive Comparative Analysis. Machine Vision and Applications Journal.
```

---

