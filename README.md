# Project: Modular Neural Network Implementation with Encoder and MLP

This project implements a modular neural network consisting of an Encoder and a Multi-Layer Perceptron (MLP) for image classification. It demonstrates how to combine feature extraction and classification models in PyTorch and evaluates their performance on a standard image dataset.

## Key Features

### 1. Modular Architecture
- **Encoder**:
  - Extracts latent features from input images.
  - Implements convolutional layers with ReLU activation and pooling.
- **MLP (Multi-Layer Perceptron)**:
  - Classifies the extracted features into target labels.
  - Implements fully connected layers with ReLU activation and a final output layer.

### 2. Training Pipeline
- **Loss Function**: CrossEntropyLoss for multi-class classification.
- **Optimizer**: Adam optimizer for effective gradient descent.
- **Training Loop**:
  - Combines the Encoder and MLP networks.
  - Trains the combined model with batch data from the training set.
  - Tracks the loss and prints progress.

### 3. Evaluation
- The trained model achieves **97.86% accuracy** on the test dataset.

### 4. Visualization
- Loss is plotted across epochs to monitor model convergence during training.

## Highlights
- Demonstrates the use of modular neural network design for flexibility.
- Provides a clear pipeline for training and evaluating deep learning models.
- Achieves high performance on the test dataset with a simple architecture.
