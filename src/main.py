from collections import namedtuple

import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import randperm
from config import DATA_DIR, MODELS_DIR
from src.consistency import (
    clac_cronbach_alpha,
    calc_items_kmeans,
    direct_consistency_calculation,
    mf_consistency_calculation,
)
from src.data_set import RatingsDataset
from src.loss import MiningOutliersLoss
from src.model import MF, SingleMF
from src.runner import Runner, SingleMFRunner
from src.utils import (
    create_dataset,
    mine_outliers_sklearn,
    mine_outliers_scipy,
    DataConverter,
    DataProcessor,
    mean_centralised,
)

import plotly.express as px


DF_PATH = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000.csv"
)


def select_n_random(trainset: RatingsDataset):
    """
    Selects n random datapoints and their corresponding labels from a dataset
    """
    perm = randperm(len(trainset))
    return trainset[perm][:100]


def plot_item_clustering(model: MF, data_converter: DataConverter):
    Row = namedtuple("Row", ["item_id", "cluster"])
    item_kmeans = calc_items_kmeans(model=model)

    item_clustering_df = []
    items = data_converter.encoded_df.item_id.unique()
    for item in items:
        item_id = data_converter.get_original_item_id(encoded_id=item)
        cluster = item_kmeans.labels_[item]
        item_clustering_df.append(Row(item_id=item_id, cluster=cluster))

    item_df = pandas.DataFrame(item_clustering_df)

    fig = px.scatter(
        item_df, x="item_id", y="cluster", color="cluster", size="cluster", hover_data=["item_id"]
    )
    fig.show()


if __name__ == "__main__":
    columns = ["workerID", "SongId", "Valence"]
    original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    original_df.columns = ["user_id", "item_id", "rating"]

    data_converter = DataConverter(original_df=original_df)
    valence_model = MF(n_users=data_converter.n_users, n_items=data_converter.n_item, include_bias=True)
    valence_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/raw/valence.pt"))

    mf_consistency_calculation(data_frame=original_df, model=valence_model)
