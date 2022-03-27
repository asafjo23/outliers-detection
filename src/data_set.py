from typing import Tuple

from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    def __init__(
        self,
        users_tensor: LongTensor,
        items_tensor: LongTensor,
        ratings_tensor: FloatTensor,
    ):
        self.users_tensor = users_tensor
        self.items_tensor = items_tensor
        self.ratings_tensor = ratings_tensor

    def __getitem__(self, index: int) -> Tuple:
        return (
            self.users_tensor[index],
            self.items_tensor[index],
            self.ratings_tensor[index],
        )

    def __len__(self) -> int:
        return self.users_tensor.size(0)
