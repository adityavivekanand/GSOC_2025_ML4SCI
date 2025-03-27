# Specific Task 4: Diffusion Models with DDPM and U-Net

This project implements a diffusion model using a Denoising Diffusion Probabilistic Model (DDPM) framework with a U-Net based architecture. The codebase is modularized into separate files for the model definition, utility functions, training, and sampling/inference.

## Diffusion Models and DDPM

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
