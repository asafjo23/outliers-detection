from collections import namedtuple

import numpy as np
import pandas
from matplotlib import pyplot as plt
from pandas import read_csv
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import randperm
from config import DATA_DIR
from src.consistency import clac_cronbach_alpha, calc_items_kmeans, direct_calculation
from src.data_set import RatingsDataset
from src.loss import MiningOutliersLoss
from src.model import MF
from src.runner import Runner
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

    direct_calculation(data_frame=original_df)
    data_converter = DataConverter(
        original_df=original_df, n_random_users=10, n_ratings_per_random_user=9
    )

    valence_model = MF(n_users=data_converter.n_users, n_items=data_converter.n_item,)
    epochs = 1

    criterion = MSELoss()
    optimizer = SGD(valence_model.parameters(), lr=5, weight_decay=1e-3)
    runner = Runner(model=valence_model, criterion=criterion, optimizer=optimizer, epochs=epochs)

    train_set = create_dataset(data_converter=data_converter)
    train_load = DataLoader(train_set, batch_size=1000, shuffle=True)
    users, items, ratings = select_n_random(train_set)

    with SummaryWriter("runs/dev") as writer:
        writer.add_graph(valence_model, (users, items))

        for epoch in range(epochs):
            epoch_loss = runner.train(train_loader=train_load, epoch=epoch, writer=writer)
            print(f"epoch={epoch + 1}, loss={epoch_loss}")

    direct_calculation(data_frame=original_df)
    # plot_item_clustering(model=valence_model, data_converter=data_converter)

    # columns = ["user_id", "item_id", "rating"]
    # original_df = read_csv(
    #     DF_PATH, skipinitialspace=True, sep=";", names=columns, encoding="latin-1", low_memory=False
    # )
    # data_converter = DataConverter(
    #     original_df=original_df, n_random_users=0, n_ratings_per_random_user=200
    # )
    #
    # model = MF(
    #     n_users=data_converter.n_users,
    #     n_items=data_converter.n_item,
    # )
    #
    # criterion = MSELoss()
    # optimizer = SGD(model.parameters(), lr=5, weight_decay=1e-5)
    # epochs = 5
    #
    # runner = Runner(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     epochs=epochs
    # )
    #
    # train_set = create_dataset(data_converter=data_converter)
    # train_load = DataLoader(train_set, batch_size=1000, shuffle=True)
    # with SummaryWriter("runs/Book-Crossing") as writer:
    #     for epoch in range(epochs):
    #         epoch_loss = runner.train(train_loader=train_load, epoch=epoch, writer=writer)
    #         print(f"epoch={epoch + 1}, loss={epoch_loss}")
    #
    # outliers = mine_outliers_sklearn(model=model, data_converter=data_converter)
