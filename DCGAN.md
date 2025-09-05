---
license: mit
datasets:
- fka/awesome-chatgpt-prompts
language:
- en
metrics:
- accuracy
base_model:
- nanonets/Nanonets-OCR-s
new_version: moonshotai/Kimi-K2-Instruct
pipeline_tag: text-classification
library_name: adapter-transformers
tags:
- code
---


# 🧠 DCGAN: Generate Realistic Human Faces using PyTorch

This project uses **Deep Convolutional GAN (DCGAN)** architecture implemented in **PyTorch** to generate realistic human face images from a noise vector. It includes full training and image generation pipeline.

---

## 🔧 Features

- Uses `torchvision.datasets.ImageFolder` to load real face images.
- Implements a DCGAN:
  - `Generator` using `ConvTranspose2d` layers.
  - `Discriminator` using `Conv2d` layers without Sigmoid.
- Trains on 64x64 images.
- Supports CUDA (GPU) acceleration.
- Saves sample images every 10 epochs.
- Includes function to generate and display one random face at the end.

---

## 📁 Folder Structure


```cmd
project/
│
├── images/ # Folder with real training face images
│ ├── class_x/ # ImageFolder requires class subdirectories
│ ├── image1.jpg
│ ├── image2.jpg
│
├── generated_epoch_10.png # Sample outputs saved every 10 epochs
├── generator.pth # Saved Generator weights after training
├── gan_faces.py # Python script with model and training code
└── README.md # This file
```

---

## 🚀 Getting Started

### ✅ Prerequisites

Install Python and required libraries:

```bash
pip install torch torchvision matplotlib tqdm
```
Make sure you have a folder ./images/ structured like:

```markdown
images/
└── faces/
    ├── img1.jpg
    ├── img2.jpg
```
Where faces/ is any class name (required by ImageFolder).

## 🧠 How It Works
- Generator: Takes a random noise vector and generates a 64x64x3 image.

- Discriminator: Takes a real or fake image and returns a score (real or fake).

- Loss Function: BCEWithLogitsLoss is used for both Generator and Discriminator.

- Label Smoothing: Real images use soft labels (0.9 instead of 1.0) to stabilize training.

## ⚙️ Training
To start training:

```bash
python gan_faces.py
```
The script will:

- Load images from ./images/

- Train for 100 epochs (can be changed)

- Save sample images every 10 epochs as generated_epoch_X.png

- Save final model as generator.pth

## 🧪 Generate One Face
At the end of training, a sample face will be displayed using matplotlib.

- You can also use this standalone function:

```python
generate_one_face("generator.pth")
```
To generate and visualize a face anytime.
