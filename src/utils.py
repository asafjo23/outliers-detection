import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from torch import Tensor, sum, LongTensor, FloatTensor, histc
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
from typing import Tuple
from torch.nn import Parameter
from src.data_set import RatingsDataset
from src.model import MF


class ProcColumn:
    def __init__(self, column: Series):
        uniq = column.unique()
        self._name_2_index = {o: i for i, o in enumerate(uniq)}
        self._idx_2_name = {i: e for i, e in enumerate(self._name_2_index.keys())}

        self.encoded_col = np.array([self._name_2_index[x] for x in column])

    def get_index(self, name: str) -> int:
        return self._name_2_index[name]

    def get_name(self, index: int) -> str:
        return self._idx_2_name[index]


class DataProcessor:
    def __init__(self, original_df: DataFrame):
        (
            self.ratings_by_user,
            self.histograms_by_users,
            self.item_to_index_rating,
        ) = DataProcessor.data_process(original_df=original_df)

    @staticmethod
    def data_process(original_df: DataFrame) -> Tuple:
        """
        This function creates the original ratings embedding for each user and saves mapping
        from index to item place in the rating.
        In addition, it also creates the original histogram of the ratings of the user.
        :param original_df: original dataframe
        :return: Tuple of ratings_by_users, histograms_by_users, item_to_index_rating
        """
        ratings_by_users, histograms_by_users, item_to_index_rating = {}, {}, {}
        items_grouped_by_users = original_df.groupby("user_id")

        for user_id, group in items_grouped_by_users:
            ratings_as_tensor = Tensor(group.rating.values)
            ratings_by_users[user_id] = Parameter(ratings_as_tensor, requires_grad=False)
            histograms_by_users[user_id] = histc(ratings_as_tensor, bins=9, min=1, max=9)
            item_to_index_rating[user_id] = {
                row.item_id: i for i, row in enumerate(group.itertuples())
            }

        return ratings_by_users, histograms_by_users, item_to_index_rating


class DataConverter:
    def __init__(self, original_df: DataFrame, n_random_users: int, n_ratings_per_random_user: int):
        assert list(original_df.columns) == ["user_id", "item_id", "rating"]
        original_data = original_df.copy()
        if n_random_users > 0:
            random_users = DataConverter.create_random_users(
                original_df=original_data,
                number_of_users_to_add=n_random_users,
                n_ratings_per_random_user=n_ratings_per_random_user,
            )
            original_data = pd.concat([original_data, random_users], ignore_index=True)

        self._user_original_id_to_encoded_id = ProcColumn(original_data.user_id)
        self._item_original_id_to_encoded_id = ProcColumn(original_data.item_id)
        self.original_df = original_data
        self.encoded_df = original_data.copy()
        self.encoded_df.user_id = self._user_original_id_to_encoded_id.encoded_col
        self.encoded_df.item_id = self._item_original_id_to_encoded_id.encoded_col

        self.n_users = self.original_df.user_id.nunique()
        self.n_item = self.original_df.item_id.nunique()

    def get_original_user_id(self, encoded_id: int) -> str:
        return self._user_original_id_to_encoded_id.get_name(index=encoded_id)

    def get_original_item_id(self, encoded_id: int) -> str:
        return self._item_original_id_to_encoded_id.get_name(index=encoded_id)

    def get_encoded_user_ids(self) -> np.ndarray:
        return self._user_original_id_to_encoded_id.encoded_col

    def get_encoded_item_ids(self) -> np.ndarray:
        return self._item_original_id_to_encoded_id.encoded_col

    @staticmethod
    def create_random_users(
        original_df: DataFrame, number_of_users_to_add: int, n_ratings_per_random_user: int
    ) -> DataFrame:
        assert list(original_df.columns) == ["user_id", "item_id", "rating"]
        Row = namedtuple("Row", ["user_id", "item_id", "rating"])
        random_data = []
        original_num_of_users = original_df.user_id.nunique()
        for i in range(original_num_of_users, original_num_of_users + number_of_users_to_add):
            for _ in range(n_ratings_per_random_user):
                random_song_id = np.random.choice(original_df.item_id.values)
                random_rating = np.random.randint(1, 9)
                random_data.append(
                    Row(
                        user_id=f"random_guy_{i}",
                        item_id=random_song_id,
                        rating=random_rating,
                    )
                )

        return DataFrame(random_data, columns=["user_id", "item_id", "rating"])


def create_dataset(data_converter: DataConverter):
    users_tensor = LongTensor(data_converter.encoded_df.user_id.values)
    items_tensor = LongTensor(data_converter.encoded_df.item_id.values)
    ratings_tensor = FloatTensor(data_converter.encoded_df.rating.values)

    return RatingsDataset(
        users_tensor=users_tensor,
        items_tensor=items_tensor,
        ratings_tensor=ratings_tensor,
    )


def l2_regularize(array) -> Tensor:
    loss = sum(array**2.0)
    return loss


def mine_outliers(model: MF, data_converter: DataConverter):
    optimized_user_embeddings = np.array(model.user_factors.weight.data)
    c_similarity = cosine_similarity(optimized_user_embeddings)
    similarities = c_similarity.sum(axis=1)
    c_similarity_scores = {
        data_converter.get_original_user_id(i): score for i, score in enumerate(similarities)
    }
    dists = dict(sorted(c_similarity_scores.items(), key=lambda item: item[1]))

    items_group_by_users = data_converter.original_df.groupby("user_id")
    for user_id, item_id in dists.items():
        number_of_items = len(items_group_by_users.get_group(user_id))
        print(f"user: {user_id}, dist: {item_id}, #items: {number_of_items}")
