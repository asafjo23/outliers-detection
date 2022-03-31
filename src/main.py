from collections import namedtuple
from enum import Enum

import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.manifold import TSNE
from torch.nn import MSELoss
from torch.optim import SGD, RMSprop
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import randperm
from config import DATA_DIR, MODELS_DIR
from src.consistency import (
    clac_cronbach_alpha,
    calc_items_kmeans,
    direct_consistency_calculation,
    mf_consistency_calculation,
    lower_embeddings_dims,
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
    mean_normalized,
    ProcColumn,
)

import plotly.express as px

DF_PATH = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000_standardized.csv"
)

Row = namedtuple("Row", "workerID SongId Valence Arousal Emotion")


def select_n_random(trainset: RatingsDataset):
    """
    Selects n random data points and their corresponding labels from a dataset
    """
    perm = randperm(len(trainset))
    return trainset[perm][:100]


def get_emotion(valence: int, arousal: int) -> str:
    """
    Selects emotion based on Valence/Arousal.
    """
    if arousal <= 5 and valence <= 5:
        return "Low Negative"

    if arousal >= 5 and valence <= 5:
        return "High Negative"

    if arousal <= 5 and valence >= 5:
        return "Low Positive"

    if arousal >= 5 and valence >= 5:
        return "High Positive"


if __name__ == "__main__":
    columns = ["workerID", "SongId", "Valence"]
    valence_original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    valence_original_df.columns = ["user_id", "item_id", "rating"]

    valence_data_converter = DataConverter(
        original_df=valence_original_df, n_random_users=0, n_ratings_per_random_user=0
    )

    columns = ["workerID", "SongId", "Arousal"]
    arousal_original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    arousal_original_df.columns = ["user_id", "item_id", "rating"]

    arousal_data_converter = DataConverter(
        original_df=arousal_original_df, n_random_users=0, n_ratings_per_random_user=0
    )

    valence_model = MF(
        n_users=valence_data_converter.n_users,
        n_items=valence_data_converter.n_item,
        include_bias=True,
    )

    arousal_model = MF(
        n_users=arousal_data_converter.n_users,
        n_items=arousal_data_converter.n_item,
        include_bias=True,
    )

    epochs = 100

    criterion = MSELoss()
    optimizer = SGD(valence_model.parameters(), lr=5, weight_decay=1e-3)
    runner = Runner(
        model=valence_model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs
    )

    train_set = create_dataset(data_frame=valence_data_converter.encoded_df)
    train_load = DataLoader(train_set, batch_size=1000, shuffle=True)
    users, items, ratings = select_n_random(train_set)

    with SummaryWriter("runs/DEAM/standardized/valence") as writer:
        writer.add_graph(valence_model, (users, items))

        for epoch in range(epochs):
            epoch_loss = runner.train(train_loader=train_load, epoch=epoch, writer=writer)

            print(f"epoch={epoch + 1}, loss={epoch_loss}")

    torch.save(valence_model.state_dict(), f"{MODELS_DIR}/DEAM/standardized/valence.pt")
