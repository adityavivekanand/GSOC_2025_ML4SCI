import argparse
import torch
import torchvision
from model import DenoisingNN
from utils import LinearNoiseScheduler

def sample(args):
    model = DenoisingNN(input_dim=784, hidden_dim=256)
    scheduler = LinearNoiseScheduler(args.timesteps, args.beta_start, args.beta_end)
    
    noisy_x = torch.randn(64, 784)
    for t in reversed(range(args.timesteps)):
        pred = model(noisy_x)
        noisy_x = scheduler.add_noise(pred, torch.randn_like(pred), torch.tensor([t]))
    
    torchvision.utils.save_image(noisy_x.view(64, 1, 28, 28), "generated_samples.png")
    print("Samples saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    args = parser.parse_args()
    sample(args)