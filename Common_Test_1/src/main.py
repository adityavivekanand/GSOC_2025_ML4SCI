import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import NPYDataset
from model import get_model
from train import train_model
from test import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a ResNet model on NPY dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet10", "resnet34", "resnet50", "resnet152"],
                        help="ResNet model to use")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of output classes")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = NPYDataset(args.train_path, transform=transform)
    val_dataset = NPYDataset(args.val_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader}
    
    model = get_model(args.model, args.num_classes)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    model = train_model(model, dataloaders, criterion, optimizer, device, args.epochs)
    
    # Evaluate model
    evaluate_model(model, val_loader, device, args.num_classes)

if __name__ == "__main__":
    main()
