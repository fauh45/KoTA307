from argparse import ArgumentParser
from itertools import product
from typing import List
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr

import os
import torch

from dataset.k_fold_dataset import KFoldDataset
from dataset.pair_training_dataset import TrainingProductPairDataset
from dataset.recommendation_validation_dataset import (
    RecommendationValidationDataset,
)
from models.model_training import ModelTraining
from models.pair_embedding import PairEmbeddingModel


class Experiment:
    def __init__(
        self,
        models: List[PairEmbeddingModel],
        dataset_path: str,
        epoch: set[int],
        batch_size: set[int],
        learning_rate: set[float],
        recommendation_amount: int = 3,
        min_product_bought: int = 3,
        k_splits: int = 10,
        start_models_on: int = 0,
        start_experiment_on: int = 0,
        save_dir: str = "",
        dry_run: bool = False,
    ) -> None:
        self.__k_fold_dataset = KFoldDataset(
            dataset_path, min_product_bought, k_splits
        )

        self.recommendation_amount = recommendation_amount

        self.experiment_hparams = list(
            product(epoch, batch_size, learning_rate)
        )
        self.current_experiment_index = start_experiment_on

        self.models = models
        self.current_models_index = start_models_on

        if save_dir == "":
            self.save_dir = (
                f"./Experiment_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
            )
        else:
            self.save_dir = save_dir

        self.dry_run = dry_run

    def __get_summary_comment(self):
        current_experiment = self.__get_current_experiment()

        return f"EPOCH_{current_experiment[0]}_BATCH_SIZE_{current_experiment[1]}_LR_{current_experiment[2]}"

    def __get_summary_writer(self):
        return SummaryWriter(log_dir=self.__get_summary_save_path())

    def __get_summary_save_path(self):
        return f"{self.save_dir}/{self.__get_summary_comment()}"

    def __get_model_save_dir(self, model_name: str):
        return f"{self.__get_summary_save_path()}/{model_name}_Save"

    def __get_current_experiment(self):
        return self.experiment_hparams[self.current_experiment_index]

    def __get_complete_dataset(self):
        return self.k_fold_dataset.complete_dataset

    def __get_current_model(self):
        return self.models[self.current_models_index]

    def __create_embeddings(self, description: str):
        return self.__get_current_model().run_to_model_once(description)

    def __compare_embeddings(self, emb1, emb2):
        return cosine_distances(emb1, emb2)

    def __move_model_index_forward(self):
        self.current_models_index += 1

    def __reset_current_model(self):
        self.__get_current_model().model_reset()

    def __move_experiment_index_forward(self):
        self.current_experiment_index += 1

    def train(self, dataset: TrainingProductPairDataset):
        current_model = self.__get_current_model()
        hparams = self.__get_current_experiment()

        print(
            f"Training {current_model.model_name}, using Epoch {hparams[0]} Batch Size {hparams[1]} LR {hparams[2]}"
        )

        training_model = ModelTraining(
            model=current_model,
            model_epoch=hparams[0],
            dataset_batch_size=hparams[1],
            model_lr=hparams[2],
            dataset=dataset,
            summary_writer=self.__get_summary_writer(),
            model_save_path=self.__get_model_save_dir(
                current_model.model_name
            ),
            dry_run=self.dry_run,
        )

        training_model.train()

    def validate(self, dataset: RecommendationValidationDataset):
        current_model = self.__get_current_model()
        current_model.eval()
        current_model.model_eval()

        hparams = self.__get_current_experiment()

        all_unique_item = self.__k_fold_dataset.unique_items
        all_unique_item["embeddings"] = all_unique_item[
            "Product description"
        ].apply(self.__create_embeddings)

        def get_embeddings(sku: str):
            return all_unique_item[all_unique_item["Lineitem sku"] == sku][
                "embeddings"
            ].values.tolist()[0]

        validation_data = DataLoader(dataset, batch_size=hparams[1])

        with torch.no_grad():
            total_corr = 0
            total_items = 0

            for batch in validation_data:
                for seed, label in batch:
                    seed_embeddings = get_embeddings(seed["Lineitem sku"])

                    embeddings_distance = (
                        all_unique_item["embeddings"]
                        .apply(
                            lambda emb2: self.__compare_embeddings(
                                seed_embeddings, emb2
                            )
                        )
                        .sort_values(ascending=True)
                        .head(self.recommendation_amount)
                    )

                    selected_items = all_unique_item[embeddings_distance.index]

                    corr = 0
                    # Edge cased, where label might be more than selected items
                    # But len(selected_items) <= len(label), which one to choose?
                    for i, item in enumerate(selected_items):
                        corr += pearsonr(
                            item["embeddings"], label[i]["embeddings"]
                        )

                    total_corr += corr
                    total_items += len(selected_items)

            total_corr /= total_items

            print(
                f"Validation {current_model.model_name}, using Epoch {hparams[0]} Batch Size {hparams[1]} LR {hparams[2]} Total correlation {total_corr}"
            )

            return total_corr

    def run_one_experiment(self):
        summary_writer = self.__get_summary_writer()

        for train_dataset, validate_dataset in self.__k_fold_dataset:
            self.train(train_dataset)
            total_corr = self.validate(validate_dataset)

            hparams = self.__get_current_experiment()
            model_name = self.__get_current_model().model_name

            summary_writer.add_hparams(
                {
                    "model": model_name,
                    "epoch": hparams[0],
                    "batch_size": hparams[1],
                    "lr": hparams[2],
                },
                {"hparam/corr": total_corr},
                run_name=os.path.dirname(
                    os.path.realpath(__file__)
                    + os.sep
                    + self.__get_summary_save_path()
                ),
            )

    def run_experiment(self):
        for _ in range(len(self.models)):
            for _ in range(len(self.experiment_hparams)):
                self.run_one_experiment()

                self.__move_experiment_index_forward()
                self.__reset_current_model()

                if self.dry_run:
                    break

            self.__move_model_index_forward()
