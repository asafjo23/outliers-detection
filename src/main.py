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
    f"song_level/static_annotations_songs_1_2000.csv"
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
    columns = ["workerID", "SongId", "Valence", "Arousal"]
    original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    print(original_df.Valence.mean())


