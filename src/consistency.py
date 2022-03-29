import numpy as np
import pingouin as pg
import torch
from pandas import DataFrame
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook

from src.model import MF
from src.utils import DataConverter, create_dataset


def clac_cronbach_alpha(data_frame: DataFrame):
    print(pg.cronbach_alpha(data=data_frame))
    print("hey")


def calc_items_kmeans(model: MF) -> object:
    item_embeddings = list(model.item_factors.parameters())[0].detach().cpu()
    item_embeddings = np.array(item_embeddings)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(item_embeddings)
    return kmeans


def direct_consistency_calculation(data_frame: DataFrame) -> float:
    item_grouped_by_users = data_frame.groupby("item_id")
    tqdm_notebook.pandas()
    consistency = data_frame.progress_apply(
        lambda row: row["rating"] - item_grouped_by_users.get_group(row["item_id"]).rating.mean(),
        axis=1,
    ).sum()
    return consistency


def mf_consistency_calculation(data_frame: DataFrame, model: MF) -> float:
    consistency = 0.0
    model.eval()

    data_converter = DataConverter(original_df=data_frame)
    dataset = create_dataset(data_frame=data_converter.encoded_df)
    train_loader = DataLoader(dataset, batch_size=1)

    with torch.no_grad():
        for user, item, original_rating in tqdm(train_loader, desc="mf_calculation"):
            predicted_rating = model(users=user, items=item)
            consistency += original_rating.item() - predicted_rating.item()

    return consistency
