# Codveda-Technology-Data-Science-Internship-Level-3-Advanced-Task3
Task 3: Neural Networks with  TensorFlow/Keras

# MNIST Digit Classification using Deep Neural Network (DNN)
It focuses on building, training, and evaluating a **Deep Feedforward Neural Network (DNN)** for **handwritten digit classification** using the **MNIST dataset**.

## 📌 Project Overview
The goal of this project is to train a deep learning model that can accurately recognize digits (0–9) from handwritten grayscale images of size **28x28 pixels**.

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Matplotlib**
- **NumPy**

## 📂 Dataset
- **MNIST dataset** from `keras.datasets.mnist`
- 60,000 training images and 10,000 test images
- Each image is 28x28 pixels in grayscale

## 🔍 Steps Performed
1. **Data Loading & Normalization**
   - Loaded MNIST dataset using Keras
   - Normalized pixel values to range [0, 1] for better training performance

2. **Model Architecture**
   - Input: Flatten layer to convert 28x28 images into a 784-element vector
   - Hidden Layers:
     - Dense(128) with ReLU activation
     - Dense(64) with ReLU activation
   - Regularization: Dropout(0.2) to prevent overfitting
   - Output Layer: Dense(10) with Softmax activation for multi-class classification

3. **Model Compilation**
   - Optimizer: Adam
   - Loss Function: Sparse Categorical Crossentropy
   - Metrics: Accuracy

4. **Model Training**
   - Trained for 10 epochs
   - Batch size: 64
   - Validation split: 20% of training data

5. **Model Evaluation**
   - Evaluated on the 10,000-image test set
   - Plotted training/validation accuracy and loss curves

## 📊 Results
- **High accuracy achieved on test set**
- Good generalization with minimal overfitting

## 📈 Training Curves
The model’s accuracy and loss were monitored for both training and validation sets to ensure consistent improvement and detect overfitting.

## 🚀 Future Improvements
- Experiment with CNNs for improved accuracy
- Use data augmentation for better generalization
