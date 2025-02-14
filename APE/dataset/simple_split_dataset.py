from sklearn.model_selection import KFold

import pandas as pd

from dataset.cleaner_helper import description_cleaner
from dataset.pair_training_dataset import TrainingProductPairDataset
from dataset.recommendation_validation_dataset import (
    RecommendationValidationDataset,
)
from helper.debug import IS_DEBUG


class SimpleSplitDataset:
    def __init__(
        self,
        csv_path: str,
        min_product_bought: int = 3,
        train_val_split: int = 0.8,
        random_state: int = 69,
        save: bool = False,
    ) -> None:
        df = pd.read_csv(csv_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df["Product description"] = df["Product description"].apply(
            description_cleaner
        )

        B2B_SKU_CONDITION = (
            df["Lineitem sku"].str.startswith(("AIA", "AXA", "OUTINT"))
        ) | (
            df["Lineitem sku"].str.contains(
                r"^[^\d\W]{3}(?:CON|WED|DUK|PPB)[\w\d]{4}$"
            )
        )

        self.complete_dataset = df[~B2B_SKU_CONDITION]

        unique_buyer = self.complete_dataset["Email"].value_counts(dropna=True)

        # Only do training and validation with buyer that already bought 2 or more
        unique_buyer = unique_buyer[unique_buyer >= min_product_bought]

        Q1 = unique_buyer.quantile(0.25)
        Q3 = unique_buyer.quantile(0.75)
        IQR = Q3 - Q1

        outlier_condition = (unique_buyer < (Q1 - 1.5 * IQR)) | (
            unique_buyer > (Q3 + 1.5 * IQR)
        )
        non_outlier_unique_buyer = unique_buyer[~outlier_condition]

        self.unique_buyer = non_outlier_unique_buyer.index.to_list()
        self.training_unique_buyer = non_outlier_unique_buyer.sample(
            frac=train_val_split, random_state=random_state
        )
        self.validation_unique_buyer = non_outlier_unique_buyer.drop(
            self.training_unique_buyer.index
        )

        self.complete_dataset = self.complete_dataset[
            self.complete_dataset["Email"].isin(non_outlier_unique_buyer.index)
        ]

        self.unique_items = self.get_all_unique_items(self.complete_dataset)

        self.save = save

        if IS_DEBUG:
            print(
                "Distribution of unique buyers",
                non_outlier_unique_buyer.describe(),
            )

    def get_dataset(self):
        train_dataset = TrainingProductPairDataset(
            self.complete_dataset[
                self.complete_dataset["Email"].isin(
                    self.training_unique_buyer.index
                )
            ],
            self.unique_items,
        )
        validation_dataset = RecommendationValidationDataset(
            self.complete_dataset[
                self.complete_dataset["Email"].isin(
                    self.validation_unique_buyer.index
                )
            ]
        )

        if self.save:
            train_dataset.save("training.csv")
            validation_dataset.save("validation.csv")

        return train_dataset, validation_dataset

    @staticmethod
    def get_all_unique_items(cleaned_data: pd.DataFrame) -> pd.DataFrame:
        return cleaned_data.drop_duplicates(subset=["Lineitem sku"])[
            ["Lineitem sku", "Product description"]
        ].set_index("Lineitem sku")
