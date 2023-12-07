import argparse
import os

import torch
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms

import warnings

# Filter out the specific UserWarning you want to suppress
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional")

from strategy.replay import Replay
from utility.CSVsave import save_results_to_csv
from utility.utils import benchmark_selction, model_selection

criterion = torch.nn.CrossEntropyLoss()  # Define your criterion

def parse_args():
    parser = argparse.ArgumentParser(description="Read variables from the terminal.")
    
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--benchmark_name", type=str, help="Benchmark name")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--latent_layer", type=int, help="Latent layer size")
    parser.add_argument("--train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--rm_size", type=int, help="Memory Size in number of samples")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--split_ratio", type=float, help="Split ratio")
    parser.add_argument("--device", type=int, help="Device number")
    
    args = parser.parse_args()
    return args

def main():
    # Parse command line arguments
    args = parse_args()

    # Create directory structure for saving results
    os.makedirs(f"results/{args.model_name}/{args.benchmark_name}/Exp_Replay/latent_{args.latent_layer}/weight_decay_{args.weight_decay}", exist_ok=True) 

    # Define file paths for saving CSV and model checkpoints
    file_name = f"results/{args.model_name}/{args.benchmark_name}/Exp_Replay/latent_{args.latent_layer}/weight_decay_{args.weight_decay}/lr_{args.lr}_epochs_{args.train_epochs}_rm_{args.rm_size}_split_{args.split_ratio}.csv"
    model_name = f"results/{args.model_name}/{args.benchmark_name}/Exp_Replay/latent_{args.latent_layer}/weight_decay_{args.weight_decay}/lr_{args.lr}_epochs_{args.train_epochs}_rm_{args.rm_size}_split_{args.split_ratio}.pth"
   
    # Save setup information to CSV file
    if file_name is not None:  
        setup = [
            ["Learning Rate", "Latent Layer", "Epochs", "Memory Elements","Weight Decay", "Split Ratio"],
            [args.lr, args.latent_layer, args.train_epochs, args.rm_size, args.weight_decay, args.split_ratio]
        ]
        save_results_to_csv(setup, file_name)
    
    # Define image transformations
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Select benchmark
    benchmark = benchmark_selction(args.benchmark_name, n_experiences=5, seed=0, return_task_id=True, train_transform=transform, eval_transform=transform)

    # Recover the train and test streams
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    eval_mb_size = 128  # Define eval_mb_size

    # Set the device as cuda, the GPU specified as default will be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    torch.manual_seed(0)

    # Define the model and the optimizer
    model = model_selection(name = args.model_name, latent_layer=args.latent_layer, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define the strategy
    exp_replay = Replay(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=21,
        mem_mb_size=107,
        train_epochs=args.train_epochs,
        eval_mb_size=eval_mb_size,
        rm_size=args.rm_size,
        split_ratio=args.split_ratio,
        patience=3,
        device=device,
        file_name = file_name,
        path = model_name
    )

    # Train the model
    exp_replay.train(train_stream, test_stream, plotting=False)

if __name__ == "__main__":
    main()
