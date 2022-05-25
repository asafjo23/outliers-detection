from typing import Mapping

import numpy as np
import pingouin as pg
import torch
from pandas import DataFrame
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
from src.model import MF
from src.utils import DataConverter, create_dataset


def clac_cronbach_alpha(data_frame: DataFrame):
    sparse_matrix = csr_matrix(
        (data_frame.user_id.nunique(), data_frame.item_id.nunique()), dtype=np.float64
    ).toarray()

    for (index, user_id, item_id, rating) in data_frame.itertuples():
        sparse_matrix[user_id][item_id] = rating
    sparse_df = DataFrame(sparse_matrix)
    return pg.cronbach_alpha(data=sparse_df)


def direct_consistency_calculation(data_frame: DataFrame) -> float:
    item_grouped_by_users = data_frame.groupby("item_id")
    tqdm_notebook.pandas()
    consistency = data_frame.progress_apply(
        lambda row: row["rating"] - item_grouped_by_users.get_group(row["item_id"]).rating.mean(),
        axis=1,
    ).sum()
    return consistency


def mf_consistency_calculation(
    data_frame: DataFrame, model: MF, outliers: Mapping, round_prediction: bool
) -> float:
    consistency = 0.0
    max_diff = 0.0
    model.eval()

    data_converter = DataConverter(original_df=data_frame)
    dataset = create_dataset(data_frame=data_converter.encoded_df)
    train_loader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for user, item, original_rating in tqdm(train_loader, desc="mf_calculation"):
            original_user_id = data_converter.get_original_user_id(encoded_id=user.item())
            if original_user_id in outliers:
                continue
            predicted_rating = model(users=user, items=item)
            if round_prediction:
                consistency += original_rating.item() - torch.round(predicted_rating).item()
                max_diff = max(
                    max_diff, original_rating.item() - torch.round(predicted_rating).item()
                )
            else:
                consistency += original_rating.item() - predicted_rating.item()

    return consistency
