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

    @staticmethod
    def permute_dataset(cleaned_data: pd.DataFrame) -> list[pd.DataFrame]:
        email_grouped = cleaned_data.groupby("Email")

        validation_dataset_permuted = []
        for _, v in email_grouped:
            temp_group = v.reset_index()

            # Take one input, and use the rest as the label
            # e.g. [p1, p2, p3], dataset would be
            # p1 (seed input) -> {p2, p3} (label)
            # p2 -> (p1, p3)
            # p3 -> (p1, p2)
            for i in range(len(temp_group)):
                val_input = temp_group.loc[i]
                val_label = temp_group.drop(i)

                validation_dataset_permuted.append([val_input, val_label])

        return validation_dataset_permuted
