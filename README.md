
# PlantNet

This repository contains a PyTorch implementation for the Plant Seedlings Classification problem. The goal is to classify images of plant seedlings into one of 12 different species based on their visual characteristics.

## Requirements

- Python 3.6 or later
- PyTorch
- scikit-learn
- matplotlib

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/plant-seedlings-classification.git
cd plant-seedlings-classification
```

2. Download the dataset and place it in the `data/train` directory. The dataset should contain images of plant seedlings, organized into subdirectories corresponding to the different species.

3. Run the training script with the desired configuration:

```bash
python train.py --model resnet50 --data_dir data/train --batch_size 64 --num_epochs 50 --learning_rate 0.001
```

The available options are:

- `--model`: The model architecture to use (squeezenet, resnet50, mobilenetv2, efficientnetb0, inceptionv3).
- `--data_dir`: The directory containing the dataset.
- `--batch_size`: The batch size for training and evaluation.
- `--num_epochs`: The number of training epochs.
- `--learning_rate`: The learning rate for the optimizer.
- `--num_classes`: The number of classes in the dataset (default: 12).
- `--test_size`: The fraction of the dataset to be used as the test set (default: 0.2).
- `--val_size`: The fraction of the dataset to be used as the validation set (default: 0.2).

4. The training process will start, and the loss, accuracy, precision, recall, and F1-score for both the training and validation sets will be logged and displayed in the console.

5. After training, the model weights will be saved in the `results` directory, along with plots showing the loss, accuracy, precision, recall, and F1-score over the training epochs.

## Code Structure

- `train.py`: The main script for training the model.
- `load_dataset.py`: Contains functions for loading and splitting the dataset.
- `net.py`: Defines the model architectures (SqueezeNet, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The dataset used in this project is the [Plant Seedlings Classification dataset](https://www.kaggle.com/c/plant-seedlings-classification) from Kaggle.