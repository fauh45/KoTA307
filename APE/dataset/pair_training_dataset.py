from itertools import permutations
import modin.pandas as pd
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
            for b, g in permutations(group["Product description"].tolist(), 2):
                # label are set to 1 on description that are not the same, because of cross selling
                # to denote that the data are positive pair (a user bought two of it)
                training_dataset_permuted.append([b, g, 1 if b != g else -1])

        return pd.DataFrame(
            training_dataset_permuted,
            columns=["description_1", "description_1", "label"],
        ).reset_index(drop=True)
