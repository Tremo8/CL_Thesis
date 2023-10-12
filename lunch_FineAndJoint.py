import os

# Define the lists of values for lr, weight_decay, and train_epochs
lr_values = [0.0001, 0.001, 0.01, 0.00001]
weight_decay_values = [0, 0.001, 0.01]
train_epochs_values = [10, 15, 20]

# Iterate through the combinations of values
for lr in lr_values:
    for weight_decay in weight_decay_values:
        for train_epochs in train_epochs_values:
            # Define the command with arguments
            cmd = f"python JointTraining.py --model_name mobilenetv2 --benchmark_name split_cifar10 --lr {lr} --latent_layer 1 --train_epochs {train_epochs} --weight_decay {weight_decay} --split_ratio 0 --device 1"
            
            # Use the subprocess module to execute the command
            os.system(cmd)
