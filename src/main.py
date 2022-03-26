from pandas import read_csv
from torch.optim import SGD
from torch.utils.data import DataLoader

from config import DATA_DIR
from src.loss import MiningOutliersLoss
from src.model import MF
from src.runner import Runner
from src.utils import create_dataset, mine_outliers, DataConverter, DataProcessor

DF_PATH = (
    f"{DATA_DIR}"
    f"/DEAM/annotations/annotations per each rater/"
    f"song_level/static_annotations_songs_1_2000.csv"
)


if __name__ == "__main__":
    columns = ["workerID", "SongId", "Valence"]
    original_df = read_csv(DF_PATH, skipinitialspace=True, usecols=columns)
    original_df.columns = ["user_id", "item_id", "rating"]

    data_converter = DataConverter(
        original_df=original_df, n_random_users=10, n_ratings_per_random_user=200
    )
    data_processor = DataProcessor(original_df=data_converter.original_df)

    model = MF(
        n_users=data_converter.n_users,
        n_items=data_converter.n_item,
    )

    criterion = MiningOutliersLoss(data_converter=data_converter, data_processor=data_processor)
    optimizer = SGD(model.parameters(), lr=5, weight_decay=1e-5)
    runner = Runner(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )

    train_set = create_dataset(data_converter=data_converter)
    train_load = DataLoader(train_set, batch_size=1000, shuffle=True)
    epochs = 75
    for epoch in range(epochs):
        epoch_loss = runner.train(train_loader=train_load)
        print(f"epoch={epoch + 1}, loss={epoch_loss}")

    mine_outliers(model=model, data_converter=data_converter)
