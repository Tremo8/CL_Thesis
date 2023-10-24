import argparse
import os

import torch
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms

from strategy.latent_replay import LatentReplay
from utility.CSVsave import save_results_to_csv
from utility.utils import benchmark_selction, model_selection

# Define other necessary variables or imports
criterion = torch.nn.CrossEntropyLoss()  # Define your criterion

def parse_args():
    parser = argparse.ArgumentParser(description="Read variables from the terminal.")
    
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--benchmark_name", type=str, help="Benchmark name")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--latent_layer", type=int, help="Latent layer size")
    parser.add_argument("--train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--rm_size_MB", type=int, help="Memory Size in MB")
    parser.add_argument("--rm_size", type=int, help="Memory Size in number of samples")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--split_ratio", type=float, help="Split ratio")
    parser.add_argument("--device", type=int, help="Device number")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    os.makedirs(f"results/{args.model_name}/{args.benchmark_name}/Latent_Replay/latent_{args.latent_layer}/weight_decay_{args.weight_decay}", exist_ok=True) 

    file_name = f"results/{args.model_name}/{args.benchmark_name}/Latent_Replay/latent_{args.latent_layer}/weight_decay_{args.weight_decay}/lr_{args.lr}_epochs_{args.train_epochs}_rm_MB_{args.rm_size_MB}_rm_{args.rm_size}_split_{args.split_ratio}.csv"
    model_name = f"results/{args.model_name}/{args.benchmark_name}/Latent_Replay/latent_{args.latent_layer}/weight_decay_{args.weight_decay}/lr_{args.lr}_epochs_{args.train_epochs}_rm_MB_{args.rm_size_MB}_rm_{args.rm_size}_split_{args.split_ratio}.pth"
   
    print(f"model_name: {args.model_name}")
    print(f"benchmark_name: {args.benchmark_name}")
    print(f"lr: {args.lr}")
    print(f"latent layer: {args.latent_layer}")
    print(f"epochs: {args.train_epochs}")
    print(f"rm_size_MB: {args.rm_size_MB}")
    print(f"rm_size: {args.rm_size}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"split_ratio: {args.split_ratio}")

    if file_name is not None:  
        setup = [
            ["Learning Rate", "Latent Layer", "Epochs", "Memory MB", "Memory Elements","Weight Decay", "Split Ratio"],
            [args.lr, args.latent_layer, args.train_epochs, args.rm_size_MB, args.rm_size, args.weight_decay, args.split_ratio]
        ]
        save_results_to_csv(setup, file_name)
    
    # PyTorch modules
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    benchmark = benchmark_selction(args.benchmark_name, n_experiences=5, seed=0, return_task_id=True, train_transform=transform, eval_transform=transform)

    # recovering the train and test streams
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    eval_mb_size = 16  # Define eval_mb_size

    # Set the current device to the GPU:index
    torch.cuda.set_device(args.device) if torch.cuda.is_available() else None

    # Set the device as cuda, the GPU specified as default will be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Device: {device}") 
    
    torch.manual_seed(0)

    # Your code
    model = model_selection(name = args.model_name, latent_layer=args.latent_layer, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    latent_replay = LatentReplay(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=21,
        replay_mb_size=107,
        train_epochs=args.train_epochs,
        eval_mb_size=eval_mb_size,
        rm_size_MB=args.rm_size_MB,
        rm_size=args.rm_size,
        manual_mb=True,
        split_ratio=args.split_ratio,
        patience=3,
        device=device,
        file_name = file_name,
        path = model_name
    )

    latent_replay.train(train_stream, test_stream, plotting=False)

if __name__ == "__main__":
    main()