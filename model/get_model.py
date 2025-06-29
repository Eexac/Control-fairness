from model.CNN import CIFAR10_CNN_Tanh, CIFAR10_CNN_Relu, MNIST_CNN_Relu, MNIST_CNN_Tanh
from model.RNN import RNN_Tanh, RNN_Relu
from model.RNN_Theory import sigmod, Tanh
from model.Group_privacy import ScatterLinear
import transformers


def get_model(algorithm, dataset_name, device):
    if algorithm == 'DPSGD':
        print(1)
        if dataset_name == 'MNIST' or dataset_name == 'FMNIST':
            model = MNIST_CNN_Relu(1)
        elif dataset_name == 'MNIST_MLP_T' or dataset_name == 'FMNIST_MLP_T':
            model = ScatterLinear(1,(28,28))
        elif dataset_name == 'MNIST_CNN_T' or dataset_name == 'FMNIST_CNN_T':
            model = MNIST_CNN_Tanh(1)
        elif dataset_name == 'CIFAR-10':
            model = CIFAR10_CNN_Relu(3)
        elif dataset_name == 'CIFAR-10_MLP_T':
            model = ScatterLinear(3,(32,32))
        elif dataset_name == 'CIFAR-10_CNN_T':
            model = CIFAR10_CNN_Tanh(3)
        elif dataset_name == 'IMDB_MLP_T':
            model = sigmod()
        elif dataset_name == 'IMDB':
            model = RNN_Relu()
    else:
        if dataset_name == 'MNIST' or dataset_name == 'FMNIST':
            model = MNIST_CNN_Tanh(1)
        elif dataset_name == 'CIFAR-10':
            model = CIFAR10_CNN_Tanh(3)
        elif dataset_name == 'IMDB':
            model = RNN_Tanh()
        elif dataset_name == 'IMDB_T':
            model = sigmod()
        elif dataset_name == 'CIFAR-10-Transformers':
            # https://github.com/lxuechen/private-transformers
            model_name_or_path = 'google/vit-base-patch16-224'
            config = transformers.AutoConfig.from_pretrained(model_name_or_path)
            config.num_labels = 10

            model = transformers.ViTForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
            )

            model.requires_grad_(False)
            model.classifier.requires_grad_(True)

        elif dataset_name == 'CIFAR-100-Transformers':
            model_name_or_path = 'google/vit-base-patch16-224'
            config = transformers.AutoConfig.from_pretrained(model_name_or_path)
            config.num_labels = 100

            model = transformers.ViTForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
            )

            model.requires_grad_(False)
            model.classifier.requires_grad_(True)

    model.to(device=device)
    return model


def get_MIA_model(algorithm, dataset_name, device):
    if algorithm == 'DPSGD':
        if dataset_name == 'MNIST' or dataset_name == 'FMNIST':
            target_model = MNIST_CNN_Relu(1)
            shadow_model = MNIST_CNN_Relu(1)
            num_classes = 10
        elif dataset_name == 'CIFAR-10':
            target_model = CIFAR10_CNN_Relu(3)
            shadow_model = CIFAR10_CNN_Relu(3)
            num_classes = 10

        elif dataset_name == 'IMDB':
            target_model = RNN_Relu()
            shadow_model = RNN_Relu()
            num_classes = 2

    else:
        if dataset_name == 'MNIST' or dataset_name == 'FMNIST':
            target_model = MNIST_CNN_Tanh(1)
            shadow_model = MNIST_CNN_Tanh(1)
            num_classes = 10

        elif dataset_name == 'CIFAR-10':
            target_model = CIFAR10_CNN_Tanh(3)
            shadow_model = CIFAR10_CNN_Tanh(3)
            num_classes = 10

        elif dataset_name == 'IMDB':
            target_model = RNN_Tanh()
            shadow_model = RNN_Tanh()
            num_classes = 2

        elif dataset_name == 'CIFAR-10-Transformers':
            # https://github.com/lxuechen/private-transformers
            model_name_or_path = 'google/vit-base-patch16-224'
            config = transformers.AutoConfig.from_pretrained(model_name_or_path)
            config.num_labels = 10

            target_model = transformers.ViTForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
            )

            target_model.requires_grad_(False)
            target_model.classifier.requires_grad_(True)

            shadow_model = transformers.ViTForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
            )

            shadow_model.requires_grad_(False)
            shadow_model.classifier.requires_grad_(True)
            num_classes = 10

        elif dataset_name == 'CIFAR-100-Transformers':
            model_name_or_path = 'google/vit-base-patch16-224'
            config = transformers.AutoConfig.from_pretrained(model_name_or_path)
            config.num_labels = 100

            target_model = transformers.ViTForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
            )

            target_model.requires_grad_(False)
            target_model.classifier.requires_grad_(True)

            shadow_model = transformers.ViTForImageClassification.from_pretrained(
                model_name_or_path,
                config=config,
                ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
            )

            shadow_model.requires_grad_(False)
            shadow_model.classifier.requires_grad_(True)
            num_classes = 100
    return target_model, shadow_model, num_classes
