from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model import MF, SingleMF
from torchviz import make_dot


class Runner:
    def __init__(self, model: MF, criterion: MSELoss, optimizer: Optimizer, epochs: int):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._plot_computational_graph = False
        self._epochs = epochs

    def train(self, train_loader: DataLoader, epoch: int, writer: SummaryWriter) -> float:
        self._model.train()
        total_epoch_loss = 0.0

        with tqdm(train_loader, position=0, leave=True, unit="batch") as tepoch:
            for users, items, original_ratings in tepoch:
                original_ratings = original_ratings.float()
                predicted_ratings = self._model(users=users, items=items,)

                mse_loss = self._criterion(original_ratings, predicted_ratings)
                writer.add_scalar("Loss/train/mse_loss", mse_loss / len(users), epoch)

                loss = mse_loss

                if self._plot_computational_graph:
                    make_dot(loss).view()

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if epoch >= self._epochs - 10:
                    writer.add_histogram(
                        tag="user_factors_gradients",
                        values=self._model.user_factors.weight.grad,
                        global_step=epoch,
                    )

                total_epoch_loss += loss.item() / len(users)
                tepoch.set_postfix(train_loss=loss.item() / len(users))
                self._plot_computational_graph = False

        writer.add_scalar("Loss/train", total_epoch_loss, epoch)
        return total_epoch_loss


class SingleMFRunner:
    def __init__(self, model: SingleMF, criterion: MSELoss, optimizer: Optimizer):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer

    def train(self, train_loader: DataLoader) -> float:
        self._model.train()
        total_epoch_loss = 0.0

        for users, items, original_ratings in train_loader:
            original_ratings = original_ratings.float()
            predicted_ratings = self._model(users=users, items=items)

            mse_loss = self._criterion(original_ratings, predicted_ratings)
            loss = mse_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_epoch_loss += loss.item() / len(users)

        return total_epoch_loss
