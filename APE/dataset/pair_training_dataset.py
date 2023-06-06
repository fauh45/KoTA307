import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from helper.debug import IS_DEBUG


class TrainingProductPairDataset(Dataset):
    def __init__(
        self, cleaned_data: pd.DataFrame, unique_product: pd.DataFrame
    ) -> None:
        super().__init__()

        self.dataset = self.permute_dataset(cleaned_data, unique_product)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # get the row, flatten the result (to 1 dimension list) and convert it to a simple list
        row = self.dataset.loc[index, :].values.flatten().tolist()

        if IS_DEBUG:
            print("Selected data", row[:2])
            print("Selected label", row[2])

        # [data, label]
        return [row[:2], row[2]]

    def save(self, filename: str = "training.csv"):
        self.dataset.to_csv(filename)

    @staticmethod
    def permute_dataset(
        cleaned_data: pd.DataFrame, unique_product: pd.DataFrame
    ) -> pd.DataFrame:
        grouped_training = cleaned_data.groupby("Email")

        training_dataset_permuted = []
        for _, group in grouped_training:
            unique_product_not_user = unique_product.loc[
                ~unique_product.index.isin(group["Lineitem sku"])
            ]["Product description"].to_numpy()
            descriptions = group["Product description"].tolist()

            random_choice = np.random.choice(
                unique_product_not_user, size=(len(descriptions) * 2)
            )
            random_choice_index = 0

            for i in range(0, len(descriptions) - 1):
                # Remove training on the same description
                if descriptions[i] != descriptions[i + 1]:
                    training_dataset_permuted.append(
                        [descriptions[i], descriptions[i + 1], 1]
                    )

                training_dataset_permuted.append(
                    [descriptions[i], random_choice[random_choice_index], -1]
                )
                training_dataset_permuted.append(
                    [descriptions[i + 1], random_choice[random_choice_index + 1], -1]
                )

                random_choice_index += 2

        return pd.DataFrame(
            training_dataset_permuted,
            columns=["description_1", "description_2", "label"],
        ).reset_index(drop=True)
