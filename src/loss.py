import torch.nn.functional as F
from tensorboardX import SummaryWriter

from torch import Tensor, clone, clip, round, abs, sub, sum, tensor
from torch.nn.modules.loss import _Loss

from src.model import GaussianHistogram
from src.utils import DataConverter, DataProcessor


class MiningOutliersLoss(_Loss):
    def __init__(self, data_converter: DataConverter, data_processor: DataProcessor):
        super(MiningOutliersLoss, self).__init__()
        self._data_converter = data_converter
        self._data_processor = data_processor

        self.min_rating = data_processor.min_rating
        self.max_rating = data_processor.max_rating

        self._gauss_histo = GaussianHistogram(
            bins=self.max_rating, min=self.min_rating, max=self.max_rating, sigma=1.5
        )

    def mse_loss(
        self,
        original_ratings: Tensor,
        predicted_ratings: Tensor,
    ) -> Tensor:
        return F.mse_loss(original_ratings, predicted_ratings)

    def histogram_loss(
        self,
        users: Tensor,
        items: Tensor,
        original_ratings: Tensor,
        predicted_ratings: Tensor,
        writer: SummaryWriter,
        epoch: int,
    ) -> Tensor:
        histogram_loss = tensor(0.0)
        for user, item, original_rating, predicted_rating in zip(
            users, items, original_ratings, predicted_ratings
        ):
            user_id = self._data_converter.get_original_user_id(encoded_id=user.item())

            original_histogram = clone(self._data_processor.histograms_by_users[user_id])
            pdf_original_histogram = self._gauss_histo(original_histogram)

            if epoch == 0:
                writer.add_histogram(
                    tag=f"{user_id}/original_histogram",
                    values=original_histogram,
                    global_step=epoch,
                )

            original_rating_index = self.to_index(rating=original_rating)
            original_mass = self._calc_histogram_mass(pdf_original_histogram, original_rating_index)

            original_histogram[original_rating_index] -= 1
            predicted_round_rating = round(predicted_rating)

            predicted_rating_index = self.to_index(rating=predicted_round_rating)
            original_histogram[predicted_rating_index] += 1

            pdf_predicted_histogram = self._gauss_histo(original_histogram)
            predicted_mass = self._calc_histogram_mass(
                pdf_predicted_histogram, predicted_rating_index
            )

            histogram_loss += abs(sub(original_mass, predicted_mass)).squeeze()

            writer.add_scalars(
                f"Loss/train/histogram_mass/{user_id}",
                {
                    "original_mass": original_mass.item(),
                    "predicted_mass": predicted_mass.item(),
                },
                epoch,
            )

            writer.add_histogram(
                tag=f"{user_id}/predicted_histogram",
                values=original_histogram,
                global_step=epoch,
            )

        histogram_loss.requires_grad = True
        return histogram_loss

    def to_index(self, rating: Tensor) -> int:
        min_index = max(self.min_rating - 1, 0)
        return int(clip(rating, min=min_index, max=self.max_rating - 1).item())

    @staticmethod
    def _calc_histogram_mass(histogram: Tensor, end: int) -> Tensor:
        area = histogram[0:end]
        if len(area) == 0:
            return Tensor([0.0])

        edge_mass = 0.5 * area[len(area) - 1]
        mass = sum(area) - edge_mass
        return mass
