from collections import namedtuple
from enum import Enum

import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
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
)

import plotly.express as px


DF_PATH = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000.csv"
)


if __name__ == "__main__":
    columns = ["workerID", "SongId", "Valence"]
    valence_original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    valence_original_df.columns = ["user_id", "item_id", "rating"]

    valence_data_converter = DataConverter(
        original_df=valence_original_df, n_random_users=0, n_ratings_per_random_user=0
    )
    clac_cronbach_alpha(data_frame=valence_data_converter.encoded_df)
    #
    # columns = ["workerID", "SongId", "Arousal"]
    # arousal_original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    # arousal_original_df.columns = ["user_id", "item_id", "rating"]
    #
    # arousal_data_converter = DataConverter(
    #     original_df=arousal_original_df, n_random_users=0, n_ratings_per_random_user=0
    # )
    #
    # valence_model = MF(
    #     n_users=valence_data_converter.n_users,
    #     n_items=valence_data_converter.n_item,
    #     include_bias=True,
    # )
    #
    # arousal_model = MF(
    #     n_users=arousal_data_converter.n_users,
    #     n_items=arousal_data_converter.n_item,
    #     include_bias=True,
    # )
    #
    # valence_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/raw/valence.pt"))
    # arousal_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/raw/arousal.pt"))
    #
    # valence_item_embeddings = list(valence_model.item_factors.parameters())[0].detach().cpu()
    # arousal_item_embeddings = list(arousal_model.item_factors.parameters())[0].detach().cpu()
    #
    # valence_original_df["valence"] = valence_original_df.apply(
    #     lambda row: "Positive" if row["rating"] >= 5 else "Negative", axis=1
    # )
    #
    # valence_item_embeddings = np.array(valence_item_embeddings)
    # tsne = lower_embeddings_dims(embeddings=valence_item_embeddings)

    # def get_emotion(valence: int, arousal: int) -> str:
    #     if arousal <= 5 and valence <= 5:
    #         return "Low Negative"
    #
    #     if arousal >= 5 and valence <= 5:
    #         return "High Negative"
    #
    #     if arousal <= 5 and valence >= 5:
    #         return "Low Positive"
    #
    #     if arousal >= 5 and valence >= 5:
    #         return "High Positive"
    #
    # columns = ["workerID", "SongId", "Valence", "Arousal"]
    # original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    # original_df["Emotion"] = original_df.apply(
    #     lambda row: get_emotion(valence=row["Valence"], arousal=row["Arousal"]), axis=1
    # )
    # tsne = TSNE(n_components=2, random_state=0)
    # features = original_df[["Valence", "Arousal"]]
    # projections = tsne.fit_transform(features)
    #
    # fig = px.scatter(
    #     projections,
    #     x=0,
    #     y=1,
    #     color=original_df.Emotion,
    #     labels={"color": "Emotion"},
    #     hover_data={"song_id": original_df.SongId.values},
    # )
    # fig.update_traces(marker_size=4)
    # fig.show()
    #
    # valence_model = MF(
    #     n_users=original_df.workerID.nunique(),
    #     n_items=original_df.SongId.nunique(),
    #     include_bias=True,
    # )
    #
    # arousal_model = MF(
    #     n_users=original_df.workerID.nunique(),
    #     n_items=original_df.SongId.nunique(),
    #     include_bias=True,
    # )
    #
    # valence_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/raw/valence.pt"))
    # arousal_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/raw/arousal.pt"))
    #
    # from pandas import DataFrame
    # from src.utils import ProcColumn
    # from collections import namedtuple
    #
    # Row = namedtuple("Row", "workerID SongId Valence Arousal Emotion")
    #
    # columns = ["workerID", "SongId", "Valence", "Arousal"]
    # original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    # user_original_id_to_encoded_id = ProcColumn(original_df.workerID)
    # item_original_id_to_encoded_id = ProcColumn(original_df.SongId)
    # original_df.workerID = user_original_id_to_encoded_id.encoded_col
    # original_df.SongId = item_original_id_to_encoded_id.encoded_col
    #
    # df_after_mf = []
    # for (index, worker_id, song_id, valence, arousal) in original_df.itertuples():
    #     user_id_as_tensor = torch.LongTensor([worker_id])
    #     item_id_as_tensor = torch.LongTensor([song_id])
    #     with torch.no_grad():
    #         valence_prediction = torch.round(valence_model(
    #             users=user_id_as_tensor, items=item_id_as_tensor,
    #         )).item()
    #         arousal_prediction = torch.round(arousal_model(
    #             users=user_id_as_tensor, items=item_id_as_tensor,
    #         )).item()
    #         emotion_predicted = get_emotion(valence=valence_prediction, arousal=arousal_prediction)
    #
    #     original_worker_id = user_original_id_to_encoded_id.get_name(index=worker_id)
    #     original_item_id = item_original_id_to_encoded_id.get_name(index=song_id)
    #     df_after_mf.append(
    #         Row(
    #             workerID=original_worker_id,
    #             SongId=original_item_id,
    #             Valence=valence_prediction,
    #             Arousal=arousal_prediction,
    #             Emotion=emotion_predicted,
    #         )
    #     )
    #
    # df_after_mf = DataFrame(
    #     df_after_mf, columns=["workerID", "SongId", "Valence", "Arousal", "Emotion"]
    # )
    # df_after_mf.to_csv(f"{DATA_DIR}/DEAM/annotations/annotations per each rater/song_level/static_annotations_songs_1_2000_after_mf.csv")
    #
    # tsne = TSNE(n_components=2, random_state=0)
    # features = df_after_mf[["Valence", "Arousal"]]
    # projections = tsne.fit_transform(features)
    #
    # fig = px.scatter(
    #     projections,
    #     x=0,
    #     y=1,
    #     color=df_after_mf.Emotion,
    #     labels={"color": "Emotion"},
    #     hover_data={"song_id": df_after_mf.SongId.values},
    # )
    # fig.update_traces(marker_size=4)
    # fig.show()

    # columns = ["user_id", "item_id", "rating"]
    # original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    #
    # valence_data_converter = DataConverter(
    #     original_df=original_df, n_random_users=0, n_ratings_per_random_user=0
    # )
    #
    # valence_model = MF(
    #     n_users=valence_data_converter.n_users,
    #     n_items=valence_data_converter.n_item,
    #     include_bias=True
    # )
    # epochs = 100
    #
    # criterion = MSELoss()
    # optimizer = RMSprop(valence_model.parameters(), lr=.01)
    # runner = Runner(
    #     model=valence_model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     epochs=epochs
    # )
    #
    # train_set = create_dataset(data_frame=valence_data_converter.encoded_df)
    # train_load = DataLoader(train_set, batch_size=1000, shuffle=True)
    # users, items, ratings = select_n_random(train_set)
    #
    # with SummaryWriter("runs/DEAM/mean_normalized/valence") as writer:
    #     writer.add_graph(valence_model, (users, items))
    #
    #     for epoch in range(epochs):
    #         epoch_loss = runner.train(train_loader=train_load, epoch=epoch, writer=writer)
    #
    #         print(f"epoch={epoch + 1}, loss={epoch_loss}")
    #
