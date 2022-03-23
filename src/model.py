from torch.nn import Embedding, Module
from torch.nn.init import xavier_normal_
from torch import arange, unsqueeze, exp, Tensor, sum
from numpy import sqrt, pi


class GaussianHistogram(Module):
    def __init__(self, bins: int, min: int, max: int, sigma: float):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (arange(bins).float() + 0.5)

    def forward(self, x):
        x = unsqueeze(x, 0) - unsqueeze(self.centers, 1)
        x = exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * sqrt(pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


class MF(Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors=50,
    ):
        super().__init__()
        self.user_factors = Embedding(n_users, n_factors)
        self.item_factors = Embedding(n_items, n_factors)
        self.user_factors.weight = xavier_normal_(self.user_factors.weight)
        self.item_factors.weight = xavier_normal_(self.item_factors.weight)

        self.user_biases = Embedding(n_users, 1)
        self.item_biases = Embedding(n_items, 1)

    def forward(
        self,
        users: Tensor,
        items: Tensor,
    ) -> Tensor:
        pred = self.user_biases(users) + self.item_biases(items)
        pred += sum((self.user_factors(users) * self.item_factors(items)), dim=1, keepdim=True)
        return pred.squeeze()
