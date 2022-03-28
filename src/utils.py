import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
from typing import Tuple, Mapping
from torch.nn import Parameter
from tqdm import tqdm

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
        self.min_rating = min(original_df.rating.values)
        self.max_rating = max(original_df.rating.values)

        (
            self.ratings_by_user,
            self.histograms_by_users,
            self.item_to_index_rating,
        ) = self.data_process(original_df=original_df)

    def data_process(self, original_df: DataFrame) -> Tuple:
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
            ratings_as_tensor = torch.Tensor(group.rating.values)
            ratings_by_users[user_id] = Parameter(ratings_as_tensor, requires_grad=False)
            histograms_by_users[user_id] = torch.histc(
                ratings_as_tensor, bins=self.max_rating, min=self.min_rating, max=self.max_rating
            )
            item_to_index_rating[user_id] = {
                row.item_id: i for i, row in enumerate(group.itertuples())
            }

        return ratings_by_users, histograms_by_users, item_to_index_rating


class DataConverter:
    def __init__(self, original_df: DataFrame, n_random_users: int, n_ratings_per_random_user: int):
        assert list(original_df.columns) == ["user_id", "item_id", "rating"]
        original_data = original_df.copy()
        self.min_rating = min(original_df.rating.values)
        self.max_rating = max(original_df.rating.values)

        if n_random_users > 0:
            random_users = self.create_random_users(
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

    def create_random_users(
        self, original_df: DataFrame, number_of_users_to_add: int, n_ratings_per_random_user: int
    ) -> DataFrame:
        assert list(original_df.columns) == ["user_id", "item_id", "rating"]
        Row = namedtuple("Row", ["user_id", "item_id", "rating"])
        random_data = []
        original_num_of_users = original_df.user_id.nunique()
        for i in range(original_num_of_users, original_num_of_users + number_of_users_to_add):
            for _ in range(n_ratings_per_random_user):
                random_song_id = np.random.choice(original_df.item_id.values)
                random_rating = np.random.randint(self.min_rating, self.max_rating)
                random_data.append(
                    Row(
                        user_id=f"random_guy_{i}",
                        item_id=random_song_id,
                        rating=random_rating,
                    )
                )

        return DataFrame(random_data, columns=["user_id", "item_id", "rating"])


def mean_centralised(dataframe: DataFrame) -> DataFrame:
    items_group_by_users = dataframe.groupby("user_id")
    normalized_data = dataframe.copy()
    with tqdm(total=normalized_data.shape[0], desc="_mean_centralised") as pbar:
        for (index, user_id, item_id, rating) in normalized_data.itertuples():
            group = items_group_by_users.get_group(user_id)
            total_sum = sum(group.rating.values)
            normalized_data.at[index, "rating"] = rating - (total_sum / len(group))
            pbar.update(1)

    return normalized_data


def mean_normalized(dataframe: DataFrame) -> DataFrame:
    items_group_by_users = dataframe.groupby("user_id")
    normalized_data = dataframe.copy()
    with tqdm(total=normalized_data.shape[0], desc="_mean_normalized") as pbar:
        for (index, user_id, item_id, rating) in normalized_data.itertuples():
            group = items_group_by_users.get_group(user_id)
            mean, std = group.rating.mean(), group.rating.std()
            normalized_data.at[index, "rating"] = (rating - mean) / std
            pbar.update(1)

    return normalized_data


def create_dataset(data_converter: DataConverter):
    users_tensor = torch.LongTensor(data_converter.encoded_df.user_id.values)
    items_tensor = torch.LongTensor(data_converter.encoded_df.item_id.values)
    ratings_tensor = torch.FloatTensor(data_converter.encoded_df.rating.values)

    return RatingsDataset(
        users_tensor=users_tensor,
        items_tensor=items_tensor,
        ratings_tensor=ratings_tensor,
    )


def mine_outliers(model: MF, data_converter: DataConverter) -> Mapping:
    optimized_user_embeddings = np.array(model.user_factors.weight.data)
    c_similarity = cosine_similarity(optimized_user_embeddings)
    similarities = c_similarity.sum(axis=1)
    c_similarity_scores = {
        data_converter.get_original_user_id(i): score for i, score in enumerate(similarities)
    }
    return c_similarity_scores


def classical_outliers_mining(data_converter: DataConverter) -> Mapping:
    """
    This function tries to identify who the outliers are by using cosine similarities between all
    users ratings vectors.
    :return mapping of outliers to score:
    """
    sparse_matrix = csr_matrix(
        (data_converter.n_users, data_converter.n_item), dtype=np.float64
    ).toarray()
    items_group_by_users = data_converter.encoded_df.groupby("user_id")

    for key, group in items_group_by_users:
        total_sum = sum(group.rating.values)
        non_zeros = len(group.rating.values)
        for (index, user_id, item_id, rating) in group.itertuples():
            sparse_matrix[user_id][item_id] = rating - total_sum / non_zeros

        sparse_matrix[key] /= total_sum

    c_similarity = cosine_similarity(sparse_matrix)
    similarities = c_similarity.sum(axis=1)
    c_similarity_scores = {
        data_converter.get_original_user_id(i): score for i, score in enumerate(similarities)
    }
    return c_similarity_scores
