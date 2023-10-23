import os

# Define the lists of values for lr, weight_decay, and train_epochs
models = ['mobilenetv2', 'mobilenetv1', 'phinet_0.8_0.75_8_downsampling']
lr_values = [0.0001, 0.00001, 0.00005]
epochs= [4, 10]
latent_layer = [1]
weight_decay_values = [0, 0.0001, 0.00001]
rm_sizes = [1500]

# Iterate through the combinations of values
for model in models:
    for lr in lr_values:
        for epoch in epochs:
            for weight_decay in weight_decay_values:
                for rm_size in rm_sizes:
                    for ll in latent_layer:
                        # Define the command with arguments
                        if epoch == 4:
                            split_ratio = 0
                        else:
                            split_ratio = 0.8
                            
                        cmd = f"python ExpReplay.py --model_name {model} --benchmark_name split_cifar10 --lr {lr} --latent_layer {ll} --train_epochs {epoch} --rm_size {rm_size} --weight_decay {weight_decay} --split_ratio {split_ratio} --device 1"
                    
                        # Use the subprocess module to execute the command
                        os.system(cmd)