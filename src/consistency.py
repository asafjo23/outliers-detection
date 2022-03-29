import numpy as np
import pingouin as pg
from pandas import DataFrame
from sklearn.cluster import KMeans
from tqdm._tqdm_notebook import tqdm_notebook

from src.model import MF


def clac_cronbach_alpha(data_frame: DataFrame):
    print(pg.cronbach_alpha(data=data_frame))
    print("hey")


def calc_items_kmeans(model: MF) -> object:
    item_embeddings = list(model.item_factors.parameters())[0].detach().cpu()
    item_embeddings = np.array(item_embeddings)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(item_embeddings)
    return kmeans


def direct_calculation(data_frame: DataFrame) -> float:
    item_grouped_by_users = data_frame.groupby("item_id")
    tqdm_notebook.pandas()
    consistency = data_frame.progress_apply(
        lambda row: row["rating"] - item_grouped_by_users.get_group(row["item_id"]).rating.mean(),
        axis=1,
    ).sum()
    return consistency
