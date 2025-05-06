# DigitTorch

## Overview

DigitTorch is a simple PyTorch project that classifies handwritten digits (0–9) using the MNIST dataset.  
it's an introduction to neural networks in general and helped me learn a lot about Pytorch and its functionality.

---

## Features

- **Simple neural network** (fully connected layers) trained on MNIST
- **Accuracy report** on the test set after training
- **Optional visualization** showing predicted digits

---

## Requirements

- Python 3.7+
- PyTorch and torchvision
- Matplotlib (for optional visual plots)

---

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/digittorch.git
cd digittorch
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Code

```bash
python main.py
```

---

## How It Works

- Loads the MNIST dataset and normalizes the images.
- Defines a simple two-layer fully connected network.
- Trains the model over 5 epochs using stochastic gradient descent (SGD).
- Prints the final accuracy on the test set.
- (Optional) Displays a few test images with their predicted labels.

---

## Why This Project?

This project is perfect for beginners like me who want hands-on experience with PyTorch.  
It focuses on clear, minimal code while still touching on key machine learning steps like data loading, model definition, training, and evaluation.

