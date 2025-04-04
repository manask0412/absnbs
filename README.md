# ML Hackathon: Detection of AI-Generated Images

## Overview

This project tackles the challenge of distinguishing between AI-generated (fake) images and real images. In response to the growing sophistication of generative models (e.g., Adobe Firefly, Stable Diffusion, Midjourney), we build a binary classification model that leverages a pre-trained EfficientNetB2 network to effectively classify images as **Real** or **Fake**. Our approach emphasizes robust performance even under adversarial perturbations.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Evaluation](#evaluation)
- [Usage Instructions](#usage-instructions)
- [Team Information](#team-information)

## Dataset

We use the CIFAKE dataset from Kaggle, which contains:
- **Training Data**: 50,000 real images and 50,000 fake images.
- **Testing Data**: 10,000 real images and 10,000 fake images.
- **Image Size**: 32x32 pixels

## Model Architecture

The model is built on EfficientNetB2 with ImageNet pre-trained weights (frozen initially). The architecture is as follows:
- **Base Model**: EfficientNetB2 (without the top classification layers).
- **Global Average Pooling**: Reduces spatial dimensions.
- **Dropout (0.5)**: Added for regularization to prevent overfitting.
- **Dense Output Layer**: Single neuron with a sigmoid activation to produce a binary classification ("Real" or "Fake").

## Training Strategy

Our training is performed in two phases:

1. **Initial Training (20 epochs)**:
   - The base model remains frozen.
   - Standard Adam optimizer is used.
   - Early stopping is implemented to restore the best weights based on validation loss.
   
2. **Fine-Tuning (10 epochs)**:
   - The base model is unfrozen (allowing further fine-tuning with a lower learning rate).
   - This phase further refines the network to improve robustness..

3. **Adversarial Robustness**:
  - In addition to standard training, the model is explicitly evaluated on adversarially perturbed images. This ensures that our solution maintains high performance even when the input images have been intentionally altered to challenge the classification process.


Random seeds are set for reproducibility across TensorFlow, NumPy, and Python's random module.

## Evaluation

The model’s performance is evaluated using:
- **Loss & Binary Accuracy**
- **Classification Report**: Includes precision, recall, and F1 Score computed on the test dataset.

## Usage Instructions

### Requirements
- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- Scikit-learn
- NumPy

### How to Run

1. **Set Up Environment**: Install the required packages by running:
   ```bash
   pip install -r requirements.txt
2. **Data Preparation**: Ensure that the CIFAKE dataset is available locally. Update the `TRAIN_PATH` and `TEST_PATH` variables in the code if needed.

3. **Training the Model**:
  Execute the Python script to train and fine-tune the model:
  ```bash
  python modeltrain.py
  ```
4. **Evaluating the Model**:
  The script automatically evaluates the model on the test dataset and plots training history for both phases.

5. **Saving the Model**:
  The fine-tuned model is saved as model_name.keras.
   ```bash
   model.save("modelv1.keras")

6. **Pre-trained Model Details**:
  Model: EfficientNetB2

  Weights: Pre-trained on ImageNet (used for feature extraction)

  Usage: The base network’s weights are initially frozen during the first training phase and then unfrozen for fine-tuning. All pre-trained components have been clearly documented in the code.

### Team Information
  Team Name: Flaming ICE

  Team Members:
  Manas Kumar Jena
  Phone: 9861307722
  Email: manaskumar.jena0412@gmail.com

  Anshuman Ray
  Phone: 6371334023
  Email: anshuflame242@gmail.com
