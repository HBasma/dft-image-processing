# DFT Image Processing

## Description
This project implements image processing in the frequency domain using the Discrete Fourier Transform (DFT) with Python and Tkinter.

## Features
- Discrete Fourier Transform (DFT)
- Inverse DFT (image reconstruction)
- Amplitude spectrum visualization
- Circular filtering (low-pass / high-pass)
- Mean filter (frequency domain)
- Gaussian filter (frequency domain)

## How it works
The image is transformed to the frequency domain using `fft2`, then centered with `fftshift`.

Filtering is applied by multiplying the frequency representation with a mask, and the final image is reconstructed using `ifft2`.

### Key concepts
- Frequency domain representation
- Circular masks for filtering
- Convolution theorem:
  
  TF(image × filter) = TF(image) × TF(filter)

## Types of filtering

### Low-pass filter
Keeps low frequencies and removes high frequencies, producing a smoother image.

### High-pass filter
Keeps high frequencies to enhance edges and fine details.

### Mean and Gaussian filters
Filters are created as masks and applied in the frequency domain.

## How to run
```bash
pip install numpy pillow matplotlib
python tp4.py
