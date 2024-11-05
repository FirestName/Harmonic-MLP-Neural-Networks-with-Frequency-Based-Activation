# Harmonic-MLP-Neural-Networks-with-Frequency-Based-Activation
A novel approach to neural network architecture that incorporates harmonic frequencies into activation functions, inspired by Fourier analysis principles.


# Repository Description

An experimental implementation of a Multi-Layer Perceptron using frequency-modulated activation functions, where each neuron operates at a different harmonic frequency. This approach draws parallels between neural network components and Fourier series, treating weights as amplitudes and introducing frequency as a structural parameter.

#README.md

# Harmonic MLP

This repository presents an experimental neural network architecture that incorporates harmonic frequencies into its activation functions. The implementation treats neural network components through the lens of Fourier analysis, where activation functions act as wave components with different frequencies following a harmonic series.

# Key Concepts

Fourier-Inspired Architecture: Each neuron in a layer operates at a different frequency following the harmonic series
Frequency-Scaled Initialization: Weight initialization scaled according to the neuron's frequency
Frequency-Aware Optimization: Gradient updates are scaled based on each neuron's frequency
Smooth Activation: Uses a modified activation function that combines sinusoidal behavior with residual connections

# Architecture Details

The network architecture implements:

Frequency-based activation functions where each neuron operates at a harmonic of the base frequency
Batch normalization and dropout for training stability
Frequency-aware weight initialization
Residual connections to improve gradient flow
Custom optimizer that scales updates based on neuron frequencies

Requirements
Copytorch
torchvision
numpy
Usage
Basic usage example:
pythonCopy# Initialize model
model = HarmonicMLP()

# Define optimizer with frequency-aware updates
base_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
freq_optimizer = FrequencyAwareOptimizer(base_optimizer, model, freq_scale_factor=0.1)

# Train model
model.train()
Model Architecture
The current implementation includes:

Input layer: 784 neurons (for MNIST)
Hidden layers: [512, 256] neurons
Output layer: 10 neurons
Batch normalization after each hidden layer
Dropout (0.2) for regularization

# Performance
On the MNIST dataset, the model typically achieves:



95% test accuracy within first few epochs

<img width="708" alt="Screenshot 2024-11-05 at 14 48 33" src="https://github.com/user-attachments/assets/c5f82dc8-73f0-4c41-b6e5-2c6aefde088a">



Fast initial convergence
Stable training behavior

# Mathematical Basis
The implementation draws parallels between neural networks and Fourier analysis:

Weights ≈ Amplitude
Bias ≈ Phase
Activation functions ≈ Wave components
Each neuron operates at a different harmonic frequency

# Contributing
Contributions are welcome! Some interesting areas to explore:

Different frequency distributions
Alternative activation functions
Applications to other datasets
Performance optimizations

# Citation
If you use this code in your research, please cite:
bibtexCopy@software{harmonic_mlp,
  title = {Harmonic MLP: Neural Networks with Frequency-Based Activation},
  year = {2024},
  author = {[Your Name]},
  url = {[Repository URL]}
}
# License
[Choose appropriate license - MIT suggested for open collaboration]

# Implementation Details

The key components of the implementation are:

HarmonicLayer

pythonCopyclass HarmonicLayer(nn.Module):
    def __init__(self, input_size, output_size, base_freq=1.0, max_freq=10.0):
        
        # Layer implementation with frequency-based activation

FrequencyAwareOptimizer

pythonCopyclass FrequencyAwareOptimizer:
    def __init__(self, optimizer, model, freq_scale_factor=0.1):
        
        # Optimizer that scales updates based on frequencies

See the source code for complete implementation details.

# Contact
https://soundcloud.com/vn820hbn20nb
