from itertools import combinations
import pandas as pd
from torch.utils.data import Dataset


class TrainingProductPairDataset(Dataset):
    def __init__(self, cleaned_data: pd.DataFrame) -> None:
        super().__init__()

        self.dataset = self.permute_dataset(cleaned_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # get the row, flatten the result (to 1 dimension list) and convert it to a simple list
        row = self.dataset.loc[index, :].values.flatten().tolist()

        # [data, label]
        return [row[:2], row[2]]

    @staticmethod
    def permute_dataset(cleaned_data: pd.DataFrame) -> pd.DataFrame:
        grouped_training = cleaned_data.groupby("Email")

        training_dataset_permuted = []
        for _, group in grouped_training:
            descriptions = group["Product description"].tolist()
            for i in range(0, len(descriptions) - 1):
                training_dataset_permuted.append(
                    [descriptions[i], descriptions[i + 1], 1]
                )

            # for i in range(len(descriptions) - 1, 0, -1):
            #     training_dataset_permuted.append(
            #         [descriptions[i], descriptions[i - 1], -1]
            #     )

        return pd.DataFrame(
            training_dataset_permuted,
            columns=["description_1", "description_1", "label"],
        ).reset_index(drop=True)
