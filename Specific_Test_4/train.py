import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from model import DenoisingNN
from utils import LinearNoiseScheduler

def train(args):
    model = DenoisingNN(input_dim=784, hidden_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearNoiseScheduler(args.timesteps, args.beta_start, args.beta_end)
    
    for epoch in range(args.epochs):
        for batch in tqdm(range(100)):
            x = torch.randn(64, 784)
            noise = torch.randn_like(x)
            t = torch.randint(0, args.timesteps, (64,))
            noisy_x = scheduler.add_noise(x, noise, t)
            
            pred = model(noisy_x)
            loss = ((pred - x) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    args = parser.parse_args()
    train(args)