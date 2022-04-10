import torch
import torch.nn.functional as F
from torch import Tensor, abs, sub
from torch.nn.modules.loss import _Loss


class MiningOutliersLoss(_Loss):
    def __init__(self):
        super(MiningOutliersLoss, self).__init__()

    def mse_loss(self, original_ratings: Tensor, predicted_ratings: Tensor,) -> Tensor:
        return F.mse_loss(original_ratings, predicted_ratings)

    def histogram_loss(
        self, original_mass: Tensor, predicted_mass: Tensor, total_mass: Tensor,
    ) -> Tensor:

        histogram_loss = (
            torch.divide(abs(sub(original_mass, predicted_mass)), total_mass).sum().squeeze()
        )

        # writer.add_scalars(
        #     f"Loss/train/histogram_mass/{user_id}",
        #     {"original_mass": original_mass.item(), "predicted_mass": predicted_mass.item(),},
        #     epoch,
        # )
        #
        # writer.add_histogram(
        #     tag=f"{user_id}/predicted_histogram", values=original_histogram, global_step=epoch,
        # )

        return histogram_loss
