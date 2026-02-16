# MNIST Digit Classifier

A deep neural network for handwritten digit recognition using the MNIST dataset. Achieves over 98% accuracy with comprehensive visualization and analysis tools.

## Overview

This project implements a fully-connected neural network to classify handwritten digits from the MNIST dataset. The model includes modern regularization techniques and provides extensive visualization of training progress, predictions, and error analysis.

## Features

- Deep neural network with batch normalization and dropout
- Early stopping and learning rate scheduling
- Comprehensive training visualization
- Confusion matrix and per-class accuracy analysis
- Prediction visualization with confidence scores
- Misclassification analysis
- Detailed classification report

## Requirements

```
tensorflow
matplotlib
seaborn
scikit-learn
numpy
```

## Installation

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy
```

## Usage

Run the notebook:

```bash
jupyter notebook MNIST_Classifier.ipynb
```

The notebook will automatically:
1. Download the MNIST dataset
2. Preprocess and normalize the data
3. Build and train the neural network
4. Generate visualizations and metrics
5. Save all output images

## Model Architecture

```
Input Layer:        784 neurons (28x28 flattened)
Dense Layer 1:      512 neurons
Batch Norm 1
ReLU Activation
Dropout:            30%
Dense Layer 2:      256 neurons
Batch Norm 2
ReLU Activation
Dropout:            20%
Dense Layer 3:      128 neurons + ReLU
Output Layer:       10 neurons + Softmax
```

Total parameters: ~500,000

## Training Configuration

- Optimizer: Adam (learning rate: 0.001)
- Loss function: Categorical crossentropy
- Batch size: 128
- Maximum epochs: 25
- Validation split: 10%
- Early stopping patience: 4 epochs
- Learning rate reduction: 0.5x on plateau

## Performance

- Test accuracy: ~98%
- Training time: 3-5 minutes (CPU/GPU dependent)
- Test samples: 10,000
- Training samples: 60,000

## Output Files

The notebook generates the following visualizations:

1. `mnist_samples.png` - Grid of 20 raw MNIST digit samples
2. `class_distribution.png` - Bar chart showing class balance in training set
3. `training_curves.png` - Accuracy and loss curves over epochs
4. `confusion_matrix.png` - 10x10 confusion matrix heatmap
5. `per_class_accuracy.png` - Bar chart of accuracy per digit
6. `predictions_grid.png` - 5x5 grid of predictions (correct/incorrect)
7. `misclassified.png` - Examples of misclassified digits

All images are saved at 150 DPI in high quality.

## Key Metrics

The notebook provides:

- Overall test accuracy and loss
- Per-class precision, recall, and F1-score
- Confusion matrix showing prediction patterns
- Most commonly confused digit pairs
- Confidence scores for predictions

## Model Insights

### Common Misclassifications

The model occasionally confuses:
- 4 and 9 (similar top curves)
- 3 and 5 (similar horizontal strokes)
- 7 and 1 (when 7 is written without crossbar)

### Regularization

- Batch normalization stabilizes training
- Dropout prevents overfitting
- Early stopping ensures best model weights

## Customization

You can modify key parameters:

```python
# Training
batch_size = 128
epochs = 25
validation_split = 0.1

# Architecture
hidden_layers = [512, 256, 128]
dropout_rates = [0.3, 0.2]

# Optimizer
learning_rate = 0.001
```

## Dataset

MNIST dataset details:
- 60,000 training images
- 10,000 test images
- Image size: 28x28 pixels
- Grayscale (0-255)
- 10 classes (digits 0-9)
- Balanced class distribution

## Requirements Note

TensorFlow 2.x is required. The notebook is compatible with both CPU and GPU execution. GPU training is significantly faster but not required.

## License

This project is provided as-is for educational purposes.

## Acknowledgments

- Yann LeCun et al. for the MNIST dataset
- TensorFlow and Keras teams
- scikit-learn for evaluation metrics
