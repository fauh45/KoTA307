from sklearn.model_selection import KFold

import pandas as pd

from dataset.pair_training_dataset import TrainingProductPairDataset
from dataset.recommendation_validation_dataset import (
    RecommendationValidationDataset,
)


class KFoldDataset:
    def __init__(
        self,
        csv_path: str,
        min_product_bought: int = 3,
        splits: int = 10,
        random_state: int = 69,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        self.complete_dataset = df

        uniquer_buyer = self.complete_dataset["Email"].value_counts(
            dropna=True
        )
        # Only do training and validation with buyer that already bought 2 or more
        uniquer_buyer = uniquer_buyer[uniquer_buyer >= min_product_bought]

        self.unique_buyer = uniquer_buyer
        self.k_fold = KFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )

    def __len__(self):
        return self.k_fold.get_n_splits()

    def __iter__(self):
        self.k_fold_split = self.k_fold.split(self.unique_buyer)

        return self

    def __next__(self):
        (train_index, validation_index) = next(self.k_fold_split)

        train_dataset = TrainingProductPairDataset(
            self.complete_dataset.loc[train_index]
        )
        validation_dataset = RecommendationValidationDataset(
            self.complete_dataset.loc[validation_index]
        )

        return train_dataset, validation_dataset
