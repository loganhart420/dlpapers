from torch.utils.data import DataLoader
from notebooks.mnist import get_data


def train(epochs: int, batch_size: int, train_dataset, eval_dataset, model):
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dl = DataLoader(eval_dataset, batch_size=batch_size // 2, shuffle=True)
