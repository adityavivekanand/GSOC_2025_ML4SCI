# GSOC_2025_ML4SCI
This repository contains my solutions for Common Task 1 and Specific Task 4 under the [DeepLense](https://arxiv.org/abs/1909.07346) Project in [ML4SCI](https://ml4sci.org/) organization. 

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
https://huggingface.co/adityavivek/resnet_models/tree/main  
Hugging Face Repo: adityavivek/resnet_models

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
| Resnet34 | 0.9147   | 0.9825  | 10     |
| Resnet34 | 0.8747   | 0.9705  | 20     |
| Resnet34 | 0.8800   | 0.9782  | 30     |
| Resnet34 | 0.8614   | 0.9687  | 40     |
| Resnet50 | 0.7931   | 0.9714  | 10     |
| Resnet50 | 0.9003   | 0.9812  | 20     |
| Resnet50 | 0.9147   | 0.9814  | 30     |
| Resnet50 | 0.9226   | 0.9812  | 40     |
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