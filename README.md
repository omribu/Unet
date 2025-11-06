<!-- PROJECT TITLE + HERO -->
<!--<p align="center">
  <img src="docs/media/hero.jpg" alt="Unet" width="820">
</p> -->

<h1 align="center">Unet: Image Segmentation CNN Network</h1>

<p align="center">
A deep learning model for semantic segmentation developed as part of my M.Sc. thesis.
</p>

---

## Overview

This repository contains a Unet training model and data adjustment pipeline developed to segment plowed regions for autonomous navigation in plowed agricultural environment. 

The model is designed for line-following navigation strategy for a battery delivery robot operating in plowing fields. It performs semantic segmentation to extract binary masks from input images, enabling the robot to identify and follow crop rows.

## Key Features

- **Input Processing**: Images are resized to 512×512 pixels for network input
- **Architecture**: Classic Unet convolutional neural network
- **Output**: Binary segmentation masks for path detection
- **Application**: Agricultural robot navigation in field environments

## Project Structure
```
Unet/
├── data/              # Training and validation datasets
├── models/            # Trained model weights
├── src/               # Source code for training and inference
```

## Installation
```bash
# Clone the repository
git clone git@github.com:omribu/Unet.git
cd Unet

# Install dependencies
pip install -r requirements.txt
```

## Model Architecture

The Unet architecture consists of:
- **Encoder**: Downsampling path with convolutional and pooling layers
- **Decoder**: Upsampling path with transposed convolutions
- **Skip Connections**: Concatenating encoder features with decoder features

## Dataset

The model is trained on plowed imagery with annotated crop row segmentation masks.

## Results

#*(Add performance metrics, sample outputs, or visualization here)*
