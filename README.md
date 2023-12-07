# CL_Thesis
Latent Replay for Continual Learning on Edge devices with Efficient Architectures.

## Models
Available models:
- MobileNetV1:
  - mobilenetv1
- MobileNetV2:
  - mobilenetv2
  - 0.75_mobilenetv2
  - 0.5_mobilenetv2
- PhiNet:
  - phinet_2.3_0.75_5
  - phinet_1.2_0.5_6_downsampling
  - phinet_0.8_0.75_8_downsampling
  - phinet_1.3_0.5_7_downsampling
  - phinet_0.9_0.5_4_downsampling_deep
  - phinet_0.9_0.5_4_downsampling

## Benchmarks
Available benchmarks:
- Split CIFAR 10:
  - split_cifar10
- CORe50:
  - core50
- Split MNIST:
  - split_mnist

## Training

Expirience Replay
```bash
python ExpReplay.py --model_name phinet_0.8_0.75_8_downsampling --benchmark_name split_cifar10 --lr 0.0001 --latent_layer 1 --train_epochs 4 --rm_size 1500 --weight_decay 0 --split_ratio 0 --device 0
```

Latent Replay in Elements
```bash
python LatentReplay.py --model_name phinet_0.8_0.75_8_downsampling --benchmark_name split_cifar10 --lr 0.0001 --latent_layer 9 --train_epochs 4 --rm_size 1500 --weight_decay 0 --split_ratio 0 --device 0"
```

Latent Replay in MB
```bash
python LatentReplay.py --model_name phinet_0.8_0.75_8_downsampling --benchmark_name split_cifar10 --lr 0.0001 --latent_layer 9 --train_epochs 4 --rm_size_MB 0.5 --weight_decay 0 --split_ratio 0 --device 0"
```
