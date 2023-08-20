import os

# Define the commands with arguments
cmd1 = "python project.py --lr 0.001 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.0001 --split_ratio 0"
cmd2 = "python project.py --lr 0.0001 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.0001 --split_ratio 0"
cmd3 = "python project.py --lr 0.00005 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.0001 --split_ratio 0"
cmd4 = "python project.py --lr 0.001 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.0001 --split_ratio 0.8"
cmd5 = "python project.py --lr 0.0001 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.0001 --split_ratio 0.8"
cmd6 = "python project.py --lr 0.00005 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.0001 --split_ratio 0.8"

cmd7 = "python project.py --lr 0.001 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.001 --split_ratio 0"
cmd8 = "python project.py --lr 0.0001 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.001 --split_ratio 0"
cmd9 = "python project.py --lr 0.00005 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.001 --split_ratio 0"
cmd10 = "python project.py --lr 0.001 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.001 --split_ratio 0.8"
cmd11 = "python project.py --lr 0.0001 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.001 --split_ratio 0.8"
cmd12 = "python project.py --lr 0.00005 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.001 --split_ratio 0.8"

cmd13 = "python project.py --lr 0.001 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.01 --split_ratio 0"
cmd14 = "python project.py --lr 0.0001 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.01 --split_ratio 0"
cmd15 = "python project.py --lr 0.00005 --latent_layer 6 --train_epochs 4 --rm_size 1500 --weight_decay 0.01 --split_ratio 0"
cmd16 = "python project.py --lr 0.001 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.01 --split_ratio 0.8"
cmd17 = "python project.py --lr 0.0001 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.01 --split_ratio 0.8"
cmd18 = "python project.py --lr 0.00005 --latent_layer 6 --train_epochs 10 --rm_size 1500 --weight_decay 0.01 --split_ratio 0.8"

# Use the subprocess module to execute the commands
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)
os.system(cmd5)
os.system(cmd6)

os.system(cmd7)
os.system(cmd8)
os.system(cmd9)
os.system(cmd10)
os.system(cmd11)
os.system(cmd12)

os.system(cmd13)
os.system(cmd14)
os.system(cmd15)
os.system(cmd16)
os.system(cmd17)
os.system(cmd18)