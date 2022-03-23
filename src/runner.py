from torch.optim import Optimizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.loss import MiningOutliersLoss
from src.model import MF


class Runner:
    def __init__(
        self,
        model: MF,
        criterion: MiningOutliersLoss,
        optimizer: Optimizer,
    ):
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer

    def train(self, train_loader: DataLoader) -> float:
        self._model.train()
        total_epoch_loss = 0.0

        with tqdm(train_loader, position=0, leave=True, unit="batch") as tepoch:
            for users, items, original_ratings in tepoch:
                original_ratings = original_ratings.float()
                predicted_ratings = self._model(
                    users=users,
                    items=items,
                )

                mse_loss = self._criterion.mse_loss(
                    user_factors=self._model.user_factors,
                    item_factors=self._model.item_factors,
                    original_ratings=original_ratings,
                    predicted_ratings=predicted_ratings,
                )

                histogram_loss = self._criterion.histogram_loss(
                    users=users,
                    items=items,
                    original_ratings=original_ratings,
                    predicted_ratings=predicted_ratings,
                )

                loss = mse_loss + histogram_loss

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                total_epoch_loss += loss.item()
                tepoch.set_postfix(train_loss=loss.item())

        return total_epoch_loss
