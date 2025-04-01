**Autoencoders and RNN Implementation**

CRN23848 Neural Network Deep Learning Home Assignment 3 Student Information

Name: Venkata nanda krishna yaram 700765514

**Project Overview**

This repository contains implementations of various deep learning models using TensorFlow and Keras:

Basic Autoencoder - Learns to reconstruct MNIST images.

Denoising Autoencoder - Removes noise from noisy images.

RNN for Text Generation - Generates text using LSTM.

Sentiment Analysis with RNN - Classifies IMDB reviews as positive or negative.

Each model is implemented as a separate Python script.

**Q1: Implementing a Basic Autoencoder**

Description:

An autoencoder is a neural network designed to compress and reconstruct data.

We train it on the MNIST dataset, compressing each image to a latent space of 32 dimensions.

We also explore different latent sizes (16, 64) to analyze reconstruction quality.

Key Components:

Encoder: Dense layers reduce dimensionality.

Decoder: Expands latent representation back to 784 pixels.

Loss Function: Binary Cross-Entropy.

Evaluation: Compares original vs. reconstructed images.

**Q2: Implementing a Denoising Autoencoder**

Description:

This extends the basic autoencoder by adding Gaussian noise (mean=0, std=0.5) to images.

The goal is to reconstruct clean images from noisy inputs.

Useful for noise removal in medical imaging, security applications, and speech enhancement.

Key Components:

Noisy Image Generation: np.random.normal().

Training: The model learns to remove noise.

Comparison: Visualization of noisy vs. denoised images.

**Q3: Implementing an RNN for Text Generation**

Description:

We use LSTMs to generate text character by character.

The dataset consists of Shakespeare Sonnets.

Temperature Scaling is used to control randomness in text generation.

Key Components:

Preprocessing: Tokenizing text into sequences.

Model: LSTM layers predict the next character.

Training: Model learns text structure and patterns.

Text Generation: Sampling characters with different temperature values.

**Q4: Sentiment Classification Using RNN**

Description:

LSTM model trained on the IMDB dataset to classify reviews as positive or negative.

Uses word embeddings and tokenization for text representation.

Evaluates model performance using accuracy, precision, recall, and F1-score.

Discusses the precision-recall tradeoff.

Key Components:

Dataset: tensorflow.keras.datasets.imdb

Text Preprocessing: Tokenization and padding.

Model: LSTM-based binary classifier.

Evaluation: Confusion matrix and classification report.
