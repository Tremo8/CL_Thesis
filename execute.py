import os

# Define the commands with arguments
cmd1 = "python project.py --lr 0.001 --latent_layer 9 --train_epochs 1 --rm_size 1500 --split_ratio 0"

# Use the subprocess module to execute the commands
os.system(cmd1)