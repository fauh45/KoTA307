from sklearn.model_selection import KFold

import pandas as pd
import wandb

from dataset.cleaner_helper import description_cleaner
from dataset.pair_training_dataset import TrainingProductPairDataset
from dataset.recommendation_validation_dataset import (
    RecommendationValidationDataset,
)


class SimpleSplitDataset:
    def __init__(
        self,
        csv_path: str,
        min_product_bought: int = 3,
        train_val_split: int = 0.8,
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
        self.training_unique_buyer = uniquer_buyer.sample(
            frac=train_val_split, random_state=random_state
        )
        self.validation_unique_buyer = uniquer_buyer.drop(
            self.training_unique_buyer.index
        )

        self.unique_items = self.get_all_unique_items(self.complete_dataset)

    def recreate_unique_from_index(self, list_of_index: list) -> list:
        return [self.unique_buyer[i] for i in list_of_index]

    def __len__(self):
        return 1

    def __iter__(self):
        self.current_index = 0

        return self

    def __next__(self):
        if self.current_index < 1:
            train_dataset = TrainingProductPairDataset(
                self.complete_dataset[
                    self.complete_dataset["Email"].isin(
                        self.training_unique_buyer.index
                    )
                ]
            )
            validation_dataset = RecommendationValidationDataset(
                self.complete_dataset[
                    self.complete_dataset["Email"].isin(
                        self.validation_unique_buyer.index
                    )
                ]
            )

            train_dataset.save("training.csv")
            validation_dataset.save("validation.csv")

            wandb.log_artifact("training.csv")
            wandb.log_artifact("validation.csv")

            self.current_index += 1

            return train_dataset, validation_dataset

        raise StopIteration

    @staticmethod
    def get_all_unique_items(cleaned_data: pd.DataFrame) -> pd.DataFrame:
        return cleaned_data.drop_duplicates(subset=["Lineitem sku"])[
            ["Lineitem sku", "Product description"]
        ].set_index("Lineitem sku")
