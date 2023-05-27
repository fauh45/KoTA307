from itertools import permutations

import pandas as pd
import numpy as np
import math

from tqdm import tqdm
from torch.utils.data import Dataset


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

        # [anchor, positive, negative]
        return row

    def save(self, filename: str = "training.csv"):
        self.dataset.to_csv(filename)

    @staticmethod
    def permute_dataset(
        cleaned_data: pd.DataFrame, unique_product: pd.DataFrame
    ) -> pd.DataFrame:
        grouped_training = cleaned_data.groupby("Email")

        training_dataset_permuted = []
        for _, group in tqdm(grouped_training, total=len(grouped_training)):
            group_len = len(group)
            unique_product_not_user = unique_product.loc[
                ~unique_product.index.isin(group["Lineitem sku"])
            ]["Product description"].to_numpy()

            num_perm = int(
                math.factorial(group_len) / math.factorial(group_len - 2)
            )
            random_choice = np.random.choice(
                unique_product_not_user, size=num_perm
            )
            random_choice_index = 0

            for b, g in permutations(
                tqdm(group["Product description"].tolist()), 2
            ):
                training_dataset_permuted.append(
                    [b, g, random_choice[random_choice_index]]
                )

                random_choice_index += 1

            # descriptions = group["Product description"].tolist()
            # for i in range(0, len(descriptions) - 1):
            #     training_dataset_permuted.append(
            #         [
            #             descriptions[i],
            #             descriptions[i + 1],
            #             unique_product.loc[
            #                 ~unique_product.index.isin(group["Lineitem sku"])
            #             ]["Product description"]
            #             .sample(1, random_state=69)
            #             .values[0],
            #         ]
            #     )

        return pd.DataFrame(
            training_dataset_permuted,
            columns=["anchor", "positive", "negative"],
        ).reset_index(drop=True)
