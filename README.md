# Control-fairness
# Contro-fairness

## Installation


To reproduce the results of DPSGD-US, set the fairness parameter from {0, 0.5, 1, 1.5, 2}, training_order from {0,1} (0 corresponds to DPSGD-FS, and 1 corresponds to FS-DPSGD), one_shot from {0,1} and batch_size_fair from {8192, 19200}.

For faster training, enable GPU acceleration by setting --device gpu. Below are example bash commands:
### MNIST
```bash
python main.py --algorithm DPSGD-US --dataset_name MNIST --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```
### FMNIST

```bash
python main.py --algorithm DPSGD-US --dataset_name FMNIST --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

### CIFAR10

```bash
python main.py --algorithm DPSGD-US --dataset_name CIFAR-10 --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

### IMDB
```bash
python main.py --algorithm DPSGD-US --dataset_name IMDB --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

To reproduce the results of DPSGD-FS, set the fairness parameter from {0, 0.5, 1, 1.5, 2}, training_order from {0,1} (0 corresponds to DPSGD-US, and 1 corresponds to US-DPSGD), clip_set from {"[0.1, 0.11, 0.12]", "[0.1, 0.11]", "[0.1, 0.12]"} and batch_size_fair from {8192, 19200}.

For faster training, enable GPU acceleration by setting --device gpu. Below are example bash commands:
### MNIST
```bash
python main.py --algorithm DPSGD-FS --dataset_name MNIST --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```
### FMNIST

```bash
python main.py --algorithm DPSGD-FS --dataset_name FMNIST --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

### CIFAR10

```bash
python main.py --algorithm DPSGD-FS --dataset_name CIFAR-10 --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

### IMDB
```bash
python main.py --algorithm DPSGD-FS --dataset_name IMDB --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```



To reproduce the privacy accounting results, use the example Bash commands provided below. The Individual_privacy_accounting.py file is located in the Fairness_Algorithms directory.:
```bash
python Individual_privacy_accounting.py
```

To reproduce the results of DP-SGD, use the example bash commands below. Set --batch_size from {8192, 19200} for different configurations.

For faster training, enable GPU acceleration by setting --device gpu.
### MNIST
```bash
python main.py --algorithm DPSGD --dataset_name MNIST --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```
### FMNIST

```bash
python main.py --algorithm DPSGD --dataset_name FMNIST --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

### CIFAR10

```bash
python main.py --algorithm DPSGD --dataset_name CIFAR-10 --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
```

### IMDB
```bash
python main.py --algorithm DPSGD --dataset_name IMDB --lr 0.5 --momentum 0.9 --fairness_parameter 1 --sigma_t 5.67 --C_t 0.1 --clip_set "[0.1, 0.11, 0.12]" --epsilon 3.0 --training_order 1 --delta 1e-5 --batch_size 8192 --batch_size_fair 19200 --one_shot 1 --device cpu
