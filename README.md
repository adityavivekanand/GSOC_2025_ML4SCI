# GSOC_2025_ML4SCI
This repository contains my solutions for Common Task 1 and Specific Task 4 under the [DeepLense](https://arxiv.org/abs/1909.07346) Project in [ML4SCI](https://ml4sci.org/) organization. 

## Project Structure
```
GSOC_2025_ML4SCI/
│── Common_Test_1/
│   ├── dataset/
│   │   ├── __MACOSX/
│   │   ├── dataset/
│   │   │   ├── train/
│   │   │   ├── val/
│   ├── src/
│   │   ├── __pycache__/
│   │   ├── dataset.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── result.py
│   │   ├── test.py
│   │   ├── train.py
│   ├── README_CT1.md
│
│── model_weights/
│   ├── CT1_weights.md
│   ├── ST4_weights.md
│
│── notebooks/
│   ├── Common_Task_1/
│   │   ├── Common_Task_1.ipynb
│   ├── Specific_Task_4/
│   │   ├── config.yaml
│   │   ├── ddpm.ipynb
│
│── Results/
│   ├── CT1_Acc_vs_Epochs.png
│   ├── CT1_ROC_vs_Epochs.png
│   ├── GSOC_CT1.jpeg
│   ├── output-video.avi
│   ├── ST4_thumbnail.png
│
│── Specific_Test_4/
│   ├── __pycache__/
│   ├── Inference/
│   ├── model.py
│   ├── README_ST4.md
│   ├── sample.py
│   ├── train.py
│   ├── utils.py
│── README.md
```

# Common Task 1: Image Classification using ResNet

## About the Task
This task trains and evaluates ResNet models on an image classification dataset stored in `.npy` format. The dataset is structured into training and validation sets, and the models are trained using PyTorch. The results compare different ResNet architectures to determine the best-performing model.

## Dataset
The dataset consists of images stored in `.npy` format, categorized into different classes. The dataset is divided into:
- **Training set**: Used for training the models.
- **Validation set**: Used to evaluate model performance during training.

Each image is a NumPy array that is loaded, transformed into a tensor, and normalized before being fed into the neural network.

## Models Used
The project uses the following ResNet architectures:
- **ResNet18**
- **ResNet34**
- **ResNet50**
- **ResNet101**

Each model's final fully connected layer is modified to match the number of output classes in the dataset.

## Training
### Arguments
The training script allows for customization via command-line arguments:
| Argument | Description |
|----------|-------------|
| `--epochs` | Number of training epochs (default: 40) |
| `--batch_size` | Batch size for training and validation (default: 256) |
| `--lr` | Learning rate for the optimizer (default: 1e-3) |
| `--model` | ResNet model to use (`resnet18`, `resnet34`, `resnet50`, `resnet101`) |
| `--train_path` | Path to training dataset |
| `--val_path` | Path to validation dataset |
| `--num_classes` | Number of output classes |

### Training Script
To train a model, run the following command:
```bash
python main.py --epochs 40 --batch_size 256 --lr 0.001 --model resnet50 --train_path /path/to/train --val_path /path/to/val --num_classes 10
```

## Evaluation
Once training is complete, the model can be evaluated using:
```bash
python test.py --model resnet50 --val_path /path/to/val --num_classes 10
```
This script computes the accuracy and ROC-AUC score of the model on the validation dataset.

## Results
### Model Weights:

The trained model weights are hosted on Hugging Face Hub: 
- **Repository Name**: `adityavivek/resnet_models`
- **Model Folder**: [Weights](https://huggingface.co/adityavivek/resnet_models/tree/main/Weights)

### Predicted Output Example:
![Classified Prediction](/Results/GSOC_CT1.jpeg)

### Individual Model Comparisons
Each model is evaluated across different epochs (10, 20, 30, 40) to analyze training trends.
| Model    | Accuracy | ROC-AUC | Epochs |
|----------|----------|---------|--------|
| Resnet18 | 87.57    | 97.73   | 10     |
| Resnet18 | 89.05    | 97.66   | 20     |
| Resnet18 | 90.6     | 97.83   | 30     |
| Resnet18 | 90.96    | 98.06   | 40     |
| Resnet34 | 91.47    | 98.25   | 10     |
| Resnet34 | 87.47    | 97.05   | 20     |
| Resnet34 | 88.00    | 97.82   | 30     |
| Resnet34 | 86.14    | 96.87   | 40     |
| Resnet50 | 79.31    | 97.14   | 10     |
| Resnet50 | 90.03    | 98.12   | 20     |
| Resnet50 | 91.47    | 98.14   | 30     |
| Resnet50 | 92.26    | 98.12   | 40     |
| Resnet101| 88.63    | 97.58   | 10     |
| Resnet101| 91.65    | 98.43   | 20     |
| Resnet101| 90.84    | 97.88   | 30     |
| Resnet101| 87.56    | 96.72   | 40     |

#### Graphical Understanding for Comparing the Accuracy and ROC-AUC of each model with respect to No. of Epochs
Graphical comparison between:
![Accuracy vs No. of Epochs](/Results/CT1_Acc_vs_Epochs.png)

Graphical comparison between:
![ROC-AUC vs No. of Epochs](/Results/CT1_ROC_vs_Epochs.png)

### Best Model Selection
A final comparison is performed between **ResNet18, ResNet34, ResNet50, and ResNet101**, selecting the best epoch for each model to determine the highest accuracy and best-performing architecture.

---
This project provides a structured approach to training deep learning models on `.npy` datasets using ResNet architectures. The results guide model selection for optimal performance in image classification tasks.


# Specific Task 4: Diffusion Models

This project implements a diffusion model using a Denoising Diffusion Probabilistic Model (DDPM) framework with a U-Net based architecture. The codebase is modularized into separate files for the model definition, utility functions, training, and sampling/inference.

## Denoising Diffusion Probabilistic Models

Diffusion models are a class of generative models that learn data distributions by gradually corrupting data with noise and then reversing the process to generate new samples. The DDPM framework (Denoising Diffusion Probabilistic Model) specifically employs a Markov chain to iteratively remove noise, enabling the generation of high-quality images.

Key points:
- **Forward Process**: Gradually adds Gaussian noise to the data over a fixed number of timesteps.
- **Reverse Process**: Learns to remove the noise step-by-step, reconstructing the original image.
- **Noise Scheduler**: Uses a linear noise schedule where beta parameters are linearly spaced between `beta_start` and `beta_end`.

## Model Architecture

The model is based on a U-Net structure enhanced with attention mechanisms. The architecture comprises three main blocks:

- **DownBlock**: Applies a series of ResNet blocks with time embedding, followed by an attention module and downsampling (using average pooling). It captures hierarchical representations by reducing the spatial dimensions.
  
- **MidBlock**: Acts as a bottleneck, integrating additional ResNet and attention layers to process the most compressed representation before upsampling.
  
- **UpBlock**: Upsamples the features and concatenates them with corresponding features from the down path (skip connections). It uses ResNet blocks with time embedding and attention layers to refine the image reconstruction.

You can visualize the model architecture by running the provided visualization code. This code generates a diagram of the network, detailing the flow through linear layers, convolutional blocks, and attention modules.

## Dataset

The dataset can be downloaded from the following link:
[Dataset](https://drive.google.com/file/d/1cJyPQzVOzsCZQctNBuHCqxHnOY7v7UiA/view?usp=sharing)

The project uses a dataset containing images stored in `.npy` format. Each `.npy` file typically represents an array of image data. The configuration file (`config.yaml`) specifies the image directory path under `dataset_params.im_path`.

## Training

Training is handled in the `train.py` script, with key parameters configured in `config.yaml`. Below is an overview of the training arguments and terminology:

| **Parameter**                      | **Description** |
|------------------------------------|----------------|
| **Learning Rate (`lr`)**           | The learning rate for the optimizer (e.g., `0.0001`). |
| **Epochs (`num_epochs`)**          | The number of full passes through the training dataset (e.g., `40`). |
| **Batch Size (`batch_size`)**      | Number of samples processed per training iteration (e.g., `8`). |
| **Timesteps (`num_timesteps`)**    | Total number of diffusion steps in the forward process (e.g., `1000`). |
| **Beta Schedule (`beta_start`, `beta_end`)** | Defines the linear schedule for noise addition. |
| **Model Parameters**               | Includes image channels (`im_channels`), image size (`im_size`), and U-Net details (down, mid, up layers, time embedding dimension, etc.). |
| **Checkpoint Name (`ckpt_name`)**  | The filename for saving and loading model weights (e.g., `ddpm_ckpt.pth`). |
| **Task Name**                      | Used to designate the output directory for saving training outputs and samples. |


These parameters are outlined in the `config.yaml` file, allowing easy modifications to the training and diffusion settings.

## Video Output

During training or sampling, the generated image samples can be saved as grids. These images can be compiled into a video (or animated GIF) to visualize the progression of image quality through the reverse diffusion process. You can use tools like `ffmpeg` to create a video from the saved sample images.

https://github.com/user-attachments/assets/55258435-e730-44db-adae-7edfff64c19f

## Trained Model Weights and Repository

The trained model weights are hosted on Hugging Face Hub. The weights can be downloaded using the utility function in `utils.py`:
- **Repository Name**: `adityavivek/diffusion_model`
- **Model Filename**: `ddpm_ckpt.pth`

You can easily retrieve the weights using the provided `download_weights()` function in `utils.py`.

## Sampling / Inference

The sampling process is handled by `sample.py` and further wrapped in `inference.py`:
- **Sampling Process**: Uses the reverse diffusion process to generate images. It iteratively removes noise from an initial random Gaussian sample.
- **Inference Script**: `inference.py` sets up the environment, downloads the model weights, loads the configuration (`config.yaml`), and runs the inference pipeline via the `infer()` function in `utils.py`.

To run inference, execute:
```bash
python inference.py
