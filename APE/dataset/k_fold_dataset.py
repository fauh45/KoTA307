from sklearn.model_selection import KFold

import modin.pandas as pd

from dataset.cleaner_helper import description_cleaner
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
        df["Product description"] = df["Product description"].apply(
            description_cleaner
        )

        self.complete_dataset = df

        uniquer_buyer = self.complete_dataset["Email"].value_counts(
            dropna=True
        )
        # Only do training and validation with buyer that already bought 2 or more
        uniquer_buyer = uniquer_buyer[uniquer_buyer >= min_product_bought]

        self.unique_buyer = uniquer_buyer.index.to_list()
        self.k_fold = KFold(
            n_splits=splits, shuffle=True, random_state=random_state
        )

        self.unique_items = self.get_all_unique_items(self.complete_dataset)

    def recreate_unique_from_index(self, list_of_index: list) -> list:
        return [self.unique_buyer[i] for i in list_of_index]

    def __len__(self):
        return self.k_fold.get_n_splits()

    def __iter__(self):
        self.k_fold_split = self.k_fold.split(self.unique_buyer)

        return self

    def __next__(self):
        (train_index, validation_index) = next(self.k_fold_split)

        train_dataset = TrainingProductPairDataset(
            self.complete_dataset[
                self.complete_dataset["Email"].isin(
                    self.recreate_unique_from_index(train_index)
                )
            ]
        )
        validation_dataset = RecommendationValidationDataset(
            self.complete_dataset[
                self.complete_dataset["Email"].isin(
                    self.recreate_unique_from_index(validation_index)
                )
            ]
        )

        return train_dataset, validation_dataset

    @staticmethod
    def get_all_unique_items(cleaned_data: pd.DataFrame) -> pd.DataFrame:
        return cleaned_data.drop_duplicates(subset=["Lineitem sku"])[
            ["Lineitem sku", "Product description"]
        ].set_index("Lineitem sku")
