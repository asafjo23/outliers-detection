import torch.nn.functional as F

from torch import Tensor, clone, clip, round, abs, sub, sum
from torch.nn import Embedding
from torch.nn.modules.loss import _Loss

from src.model import GaussianHistogram
from src.utils import l2_regularize, DataConverter, DataProcessor


class MiningOutliersLoss(_Loss):
    def __init__(self, data_converter: DataConverter, data_processor: DataProcessor):
        super(MiningOutliersLoss, self).__init__()
        self._data_converter = data_converter
        self._data_processor = data_processor
        self._gauss_histo = GaussianHistogram(bins=9, min=1, max=9, sigma=1.5)

    def mse_loss(
        self,
        user_factors: Embedding,
        item_factors: Embedding,
        original_ratings: Tensor,
        predicted_ratings: Tensor,
    ) -> Tensor:
        mse_loss = F.mse_loss(original_ratings, predicted_ratings)
        mse_loss += l2_regularize(user_factors.weight) * 1e-6
        mse_loss += l2_regularize(item_factors.weight) * 1e-6
        return mse_loss

    def histogram_loss(
        self,
        users: Tensor,
        items: Tensor,
        original_ratings: Tensor,
        predicted_ratings: Tensor,
    ) -> Tensor:
        histogram_loss = Tensor([0.0])
        for user, item, original_rating, predicted_rating in zip(
            users, items, original_ratings, predicted_ratings
        ):
            user_id = self._data_converter.get_original_user_id(encoded_id=user.item())

            original_histogram = clone(self._data_processor.histograms_by_users[user_id])
            pdf_original_histogram = self._gauss_histo(original_histogram)

            original_rating_index = int(clip(original_rating, min=0, max=8).item())
            original_mass = self._calc_histogram_mass(pdf_original_histogram, original_rating_index)

            original_histogram[original_rating_index] -= 1
            predicted_round_rating_clipped = clip(round(predicted_rating), min=0, max=8)

            predicted_rating_index = int(predicted_round_rating_clipped)
            original_histogram[predicted_rating_index] += 1

            pdf_predicted_histogram = self._gauss_histo(original_histogram)
            predicted_mass = MiningOutliersLoss._calc_histogram_mass(
                pdf_predicted_histogram, predicted_rating_index
            )

            histogram_loss += abs(sub(original_mass, predicted_mass))

        histogram_loss.requires_grad = True
        return histogram_loss

    @staticmethod
    def _calc_histogram_mass(histogram: Tensor, end: int) -> Tensor:
        area = histogram[0:end]
        if len(area) == 0:
            return Tensor([0.0])

        edge_mass = 0.5 * area[len(area) - 1]
        mass = sum(area) - edge_mass
        return mass
