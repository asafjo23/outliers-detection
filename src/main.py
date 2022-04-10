from collections import namedtuple
from enum import Enum

import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.optim import SGD, RMSprop
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch import randperm
from config import DATA_DIR, MODELS_DIR
from src.consistency import (
    clac_cronbach_alpha,
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
    ProcColumn,
)

import plotly.express as px

DF_PATH_RAW = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000_raw.csv"
)

DF_PATH_MEAN_CENTRALISED = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000_mean_centralised.csv"
)

DF_PATH_STANDARDIZED = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000_standardized.csv"
)


Row = namedtuple("Row", "workerID SongId Valence Arousal Emotion")


def get_emotion(valence: int, arousal: int, valence_mean: float, arousal_mean: float) -> int:
    """
    Selects emotion based on Valence/Arousal.
    """
    if arousal <= arousal_mean and valence <= valence_mean:
        return 3

    if arousal >= arousal_mean and valence <= valence_mean:
        return 2

    if arousal <= arousal_mean and valence >= valence_mean:
        return 4

    if arousal >= arousal_mean and valence >= valence_mean:
        return 1


def to_mf_df(
    original_df: DataFrame,
    valence_df: DataFrame,
    arousal_df: DataFrame,
    valence_model: MF,
    arousal_model: MF,
):
    valence_mean = valence_df.rating.mean()
    arousal_mean = arousal_df.rating.mean()

    original_df["Emotion"] = original_df.apply(
        lambda row: get_emotion(
            valence=row["Valence"],
            arousal=row["Arousal"],
            valence_mean=valence_mean,
            arousal_mean=arousal_mean,
        ),
        axis=1,
    )

    user_original_id_to_encoded_id = ProcColumn(original_df.workerID)
    item_original_id_to_encoded_id = ProcColumn(original_df.SongId)
    original_df.workerID = user_original_id_to_encoded_id.encoded_col
    original_df.SongId = item_original_id_to_encoded_id.encoded_col

    df = []
    for (index, worker_id, song_id, valence, arousal, emotion) in original_df.itertuples():
        user_id_as_tensor = torch.LongTensor([worker_id])
        item_id_as_tensor = torch.LongTensor([song_id])
        with torch.no_grad():
            valence_prediction = valence_model(
                users=user_id_as_tensor, items=item_id_as_tensor,
            ).item()
            arousal_prediction = arousal_model(
                users=user_id_as_tensor, items=item_id_as_tensor,
            ).item()
            emotion_predicted = get_emotion(
                valence=valence_prediction,
                arousal=arousal_prediction,
                valence_mean=valence_mean,
                arousal_mean=arousal_mean,
            )

        original_worker_id = user_original_id_to_encoded_id.get_name(index=worker_id)
        original_item_id = item_original_id_to_encoded_id.get_name(index=song_id)
        df.append(
            Row(
                workerID=original_worker_id,
                SongId=original_item_id,
                Valence=valence_prediction,
                Arousal=arousal_prediction,
                Emotion=emotion_predicted,
            )
        )

    df = DataFrame(df, columns=["workerID", "SongId", "Valence", "Arousal", "Emotion"])
    return df


def k_means(dataframe: DataFrame, title: str):
    mat = dataframe[["Valence", "Arousal"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(mat)
    scaled_features = DataFrame(scaled_features)
    scaled_features.columns = ["Valence", "Arousal"]
    scaled_features["Emotion"] = dataframe.Emotion
    km = KMeans(n_clusters=4)
    y_km = km.fit_predict(scaled_features).astype(str)
    fig = px.scatter(
        x=scaled_features["Arousal"],
        y=scaled_features["Valence"],
        color_discrete_sequence=px.colors.qualitative.G10,
        color=y_km,
        hover_data={"Item id": dataframe.SongId.values},
        title=title,
    )
    fig.show()


def get_random_color():
    import random
    hexadecimal = ["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    return hexadecimal


def plot_embeddings(valence_model: MF, arousal_model: MF, dataframe: DataFrame):
    # valence_item_embeddings = list(valence_model.item_factors.parameters())[0].detach().cpu()
    # arousal_item_embeddings = list(arousal_model.item_factors.parameters())[0].detach().cpu()
    #
    # valence_item_embeddings = np.array(valence_item_embeddings)
    # valence_tsne = TSNE(n_components=2, random_state=0).fit_transform(valence_item_embeddings)
    # km = KMeans(n_clusters=2)
    # v_y_km = km.fit_predict(valence_tsne).astype(str)
    #
    # arousal_item_embeddings = np.array(arousal_item_embeddings)
    # arousal_tsne = TSNE(n_components=2, random_state=0).fit_transform(arousal_item_embeddings)
    # a_y_km = km.fit_predict(arousal_tsne).astype(str)
    #
    # fig = px.scatter(
    #     valence_tsne,
    #     x=0,
    #     y=1,
    #     color_discrete_sequence=px.colors.qualitative.G10,
    #     color=v_y_km,
    #     labels={"color": "cluster"},
    #     hover_data={"song_id": dataframe.SongId.unique()},
    #     title="Valence Item Embeddings",
    # )
    # fig.update_traces(marker_size=8)
    # fig.show()
    #
    # fig = px.scatter(
    #     arousal_tsne,
    #     x=0,
    #     y=1,
    #     color_discrete_sequence=px.colors.qualitative.G10,
    #     color=a_y_km,
    #     labels={"color": "cluster"},
    #     hover_data={"song_id": dataframe.SongId.unique()},
    #     title="Valence Item Embeddings",
    # )
    # fig.update_traces(marker_size=8)

    valence_user_embeddings = list(valence_model.user_factors.parameters())[0].detach().cpu()
    arousal_user_embeddings = list(arousal_model.user_factors.parameters())[0].detach().cpu()

    valence_user_embeddings = np.array(valence_user_embeddings)
    valence_tsne = TSNE(n_components=2, random_state=0).fit_transform(valence_user_embeddings)

    arousal_user_embeddings = np.array(arousal_user_embeddings)
    arousal_tsne = TSNE(n_components=2, random_state=0).fit_transform(arousal_user_embeddings)

    outliers = {
        "2a6b63b7690efa2390c8d9fee11b1407",
        "ad3b997c4f2382a66e49f035cacfa682",
        "65794ea9f5122952403585a237bc5e52",
        "fd5b08ce362d855ca9152a894348130c",
        "374a5659c02e12b01db6319436f17a7d",
        "bb50b45a1874ede476874bd57e4cabb4",
        "485d8e33a731a830ef0aebd71b016d08",
        "615d836ba25132081e0ebd2182221a59",
        "da37d1548ffd0631809f7be341e4fe4d",
        "a30d244141cb2f51e0803e79bc4bd147",
    }

    color = []
    for index, user in enumerate(dataframe.workerID.unique()):
        if user not in outliers:
            color.append("regular annotator")
        else:
            color.append(user)

    fig = px.scatter(
        valence_tsne,
        x=0,
        y=1,
        color_discrete_sequence=px.colors.qualitative.G10,
        color=color,
        labels={"color": "user_id"},
        hover_data={"user_id": dataframe.workerID.unique()},
        title="Valence User Embeddings",
    )
    fig.update_traces(marker_size=12)
    fig.show()

    fig = px.scatter(
        arousal_tsne,
        x=0,
        y=1,
        color_discrete_sequence=px.colors.qualitative.G10,
        color=color,
        labels={"color": "user_id"},
        hover_data={"user_id": dataframe.workerID.unique()},
        title="Valence User Embeddings",
    )
    fig.update_traces(marker_size=12)
    fig.show()


def load_models(df_path: str, type: str):
    include_bias = type == "raw"
    columns = ["workerID", "SongId", "Valence", "Arousal"]
    original_df = pandas.read_csv(df_path, skipinitialspace=True, usecols=columns)

    n_users = original_df.workerID.nunique()
    n_items = original_df.SongId.nunique()

    valence_df = original_df[["workerID", "SongId", "Valence"]]
    valence_df.columns = ["user_id", "item_id", "rating"]

    valence_model = MF(n_users=n_users, n_items=n_items, include_bias=include_bias, n_factors=300)
    valence_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/{type}/valence.pt"))

    arousal_df = original_df[["workerID", "SongId", "Arousal"]]
    arousal_df.columns = ["user_id", "item_id", "rating"]

    arousal_model = MF(n_users=n_users, n_items=n_items, include_bias=include_bias, n_factors=300)
    arousal_model.load_state_dict(torch.load(f"{MODELS_DIR}/DEAM/{type}/arousal.pt"))

    return valence_model, arousal_model, original_df, valence_df, arousal_df


if __name__ == "__main__":
    (
        valence_model_raw,
        arousal_model_raw,
        original_df_raw,
        valence_df_raw,
        arousal_df_raw,
    ) = load_models(df_path=DF_PATH_RAW, type="raw")
    # valence_model_mean_centralised, arousal_mean_centralised = load_models(df_path=DF_PATH_MEAN_CENTRALISED, type="mean_centralised")
    # valence_model_standardized, arousal_model_standardized = load_models(df_path=DF_PATH_STANDARDIZED, type="standardized")

    df_after_mf = to_mf_df(
        original_df=original_df_raw,
        valence_df=valence_df_raw,
        arousal_df=arousal_df_raw,
        valence_model=valence_model_raw,
        arousal_model=arousal_model_raw,
    )
    #
    # valence_consistency = direct_consistency_calculation(data_frame=valence_df_raw)
    # arousal_consistency = direct_consistency_calculation(data_frame=arousal_df_raw)
    #
    # print(
    #     f"Raw data consistency with outliers according to direct calculation is: \x1b[33m{valence_consistency + arousal_consistency}\x1b[32m"
    # )
    #
    # valence_df_after_mf = df_after_mf[["workerID", "SongId", "Valence"]]
    # valence_df_after_mf.columns = ["user_id", "item_id", "rating"]
    #
    # arousal_df_after_mf = df_after_mf[["workerID", "SongId", "Arousal"]]
    # arousal_df_after_mf.columns = ["user_id", "item_id", "rating"]
    #
    # valence_consistency = direct_consistency_calculation(data_frame=valence_df_after_mf)
    # arousal_consistency = direct_consistency_calculation(data_frame=arousal_df_after_mf)
    #
    # print(
    #     f"Raw data consistency with outliers according to direct calculation is: \x1b[33m{valence_consistency + arousal_consistency}\x1b[32m"
    # )
    #
    # consistency_result = DataFrame()
    # k_means(dataframe=original_df, title="Original Dataset")
    # k_means(dataframe=df_after_mf, title="After MF Dataset")
    plot_embeddings(
        valence_model=valence_model_raw, arousal_model=arousal_model_raw, dataframe=df_after_mf
    )
