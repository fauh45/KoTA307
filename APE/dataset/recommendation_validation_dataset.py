from typing import Union
from torch.utils.data import Dataset

import pandas as pd


class RecommendationValidationDataset(Dataset):
    def __init__(self, cleaned_data: pd.DataFrame) -> None:
        super().__init__()

        self.dataset = self.permute_dataset(cleaned_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def save(self, filename: str = "validation.csv"):
        buffer = "label;ground_truth\n"

        for row in self.dataset:
            row_buffer = row[0]["Product description"] + ";"
            row_buffer += (
                "["
                + ",".join(
                    f'"{desc}"'
                    for desc in row[1]["Product description"].values
                )
                + "]\n"
            )

            buffer += row_buffer

        with open(filename, "w") as f:
            f.write(buffer)

    @staticmethod
    def permute_dataset(
        cleaned_data: pd.DataFrame,
    ) -> list[Union[pd.Series, pd.DataFrame]]:
        email_grouped = cleaned_data.groupby("Email")

        validation_dataset_permuted = []
        for _, v in email_grouped:
            email_grouped_reset_index = v.reset_index()

            # Take one input, and use the rest as the label
            # e.g. [p1, p2, p3], dataset would be
            # p1 (seed input) -> {p2, p3} (label)
            # p2 -> (p1, p3)
            # p3 -> (p1, p2)
            val_input = email_grouped_reset_index.loc[0]
            val_ground_truth = email_grouped_reset_index.drop(0).reset_index(
                drop=True
            )

            validation_dataset_permuted.append([val_input, val_ground_truth])

        # print(validation_dataset_permuted[0])
        return validation_dataset_permuted
