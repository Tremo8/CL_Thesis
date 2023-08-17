import argparse
import torch
from torch.optim import Adam
import torchvision.transforms as transforms
from avalanche.benchmarks.classic import SplitCIFAR10
from strategy.latent_replay import LatentReplay
import torchvision
from micromind import PhiNet
from avalanche.models import MobilenetV1

# Define other necessary variables or imports
criterion = torch.nn.CrossEntropyLoss()  # Define your criterion

def parse_args():
    parser = argparse.ArgumentParser(description="Read variables from the terminal.")
    print("Read variables from the terminal.")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--latent_layer", type=int, help="Latent layer size")
    parser.add_argument("--train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--rm_size_MB", type=int, help="Size in MB")
    parser.add_argument("--split_ratio", type=float, help="Split ratio")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    filename = f"results/lr_{args.lr}_latent_{args.latent_layer}_epochs_{args.train_epochs}_size_{args.rm_size_MB}MB_split_{args.split_ratio}.csv"
    model_name = f"results/models/lr_{args.lr}_latent_{args.latent_layer}_epochs_{args.train_epochs}_size_{args.rm_size_MB}MB_split_{args.split_ratio}.pth"
    print(f"lr: {args.lr}")
    print(f"latent layer: {args.latent_layer}")
    print(f"epochs: {args.train_epochs}")
    print(f"rm_size_MB: {args.rm_size_MB}")
    print(f"split_ratio: {args.split_ratio}")
    # PyTorch modules
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    split_cifar = SplitCIFAR10(n_experiences=5, seed=0, return_task_id=True, train_transform=transform, eval_transform=transform)
    print("Dataset Downloaded.")
    # recovering the train and test streams
    train_stream = split_cifar.train_stream
    test_stream = split_cifar.test_stream

    eval_mb_size = 16  # Define eval_mb_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)

    # Your code
    model5 = MobilenetV1(pretrained=True, latent_layer_num=args.latent_layer).to(device)
    optimizer5 = Adam(model5.parameters(), lr=args.lr, weight_decay=0)

    args.train_epochs

    latent_replay = LatentReplay(
        model=model5,
        optimizer=optimizer5,
        criterion=criterion,
        train_mb_size=21,
        replay_mb_size=107,
        train_epochs=args.train_epochs,
        eval_mb_size=eval_mb_size,
        rm_size_MB=args.rm_size_MB,
        manual_mb=True,
        split_ratio=args.split_ratio,
        patience=3,
        device=device,
        path=model_name
    )

    latent_replay.train(train_stream, plotting=False)
    b, c = latent_replay.test(test_stream, file_name=filename)
    a = latent_replay.get_tasks_acc()
    
if __name__ == "__main__":
    print("Start the main.")
    main()
