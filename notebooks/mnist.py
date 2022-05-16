import torch
from torchvision import datasets

from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def get_data():
    training_data = datasets.MNIST(
        root="mnist",
        train=True,
        download=True,
        transform=ToTensor()
    )

    eval_data = datasets.MNIST(
        root="mnist",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return training_data, eval_data

def visualize(training_data):
    labels_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine"
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# visualize()
