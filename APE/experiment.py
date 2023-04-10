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
import numpy as np

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
        validate_only: bool = False,
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

        if not save_dir or save_dir == "":
            self.save_dir = (
                f"Experiment_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
            )
        else:
            self.save_dir = save_dir

        self.save_dir = os.getcwd() + "/" + self.save_dir

        self.dry_run = dry_run
        self.validate_only = validate_only

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
        if self.validate_only:
            # Faster Testing
            return [[1, 2, 3]]
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
            return all_unique_item.at[sku, "embeddings"]

        validation_data = DataLoader(
            dataset, batch_size=hparams[1], collate_fn=lambda x: x
        )

        with torch.no_grad():
            corr = []
            precision = []
            recall = []
            f1_score = []

            for batch in validation_data:
                for seed, label in batch:
                    seed_embeddings = get_embeddings(seed["Lineitem sku"])

                    embeddings_distance = (
                        all_unique_item.loc[
                            all_unique_item.index != seed["Lineitem sku"]
                        ]["embeddings"]
                        .apply(
                            lambda emb2: self.__compare_embeddings(
                                seed_embeddings, emb2
                            )
                        )
                        .sort_values(ascending=False)
                        .head(len(label))
                    )

                    selected_items = all_unique_item[
                        all_unique_item.index.isin(embeddings_distance.index)
                    ]

                    n_recommended = len(label)
                    n_relevant_and_recommended = len(
                        np.intersect1d(
                            selected_items.index, label["Lineitem sku"]
                        )
                    )
                    n_relevant_not_recommended = len(
                        np.setdiff1d(
                            label["Lineitem sku"], selected_items.index
                        )
                    )
                    n_relevant = (
                        n_relevant_and_recommended + n_relevant_not_recommended
                    )

                    batch_precision = (
                        n_relevant_and_recommended / n_recommended
                    )
                    batch_recall = n_relevant_and_recommended / n_relevant

                    if (batch_recall + batch_precision) > 0:
                        batch_f1_score = (
                            2 * batch_recall * batch_precision
                        ) / (batch_recall + batch_precision)
                    else:
                        batch_f1_score = 0

                    precision.append(batch_precision)
                    recall.append(batch_recall)
                    f1_score.append(batch_f1_score)

                    label["embeddings"] = label["Lineitem sku"].apply(
                        get_embeddings
                    )

                    for i, item in enumerate(selected_items.iterrows()):
                        corr.append(
                            pearsonr(
                                item[1]["embeddings"][0],
                                label.at[i, "embeddings"][0],
                            ).statistic
                        )

                print("\nBatch done!")
                print("Avg Correlation", np.average(corr))
                print("Avg Precision", np.average(precision))
                print("Avg Recall", np.average(recall))
                print("Avg F1 Score", np.average(f1_score))
                print("\n")

            print(
                f"Validation {current_model.model_name}, using Epoch {hparams[0]} Batch Size {hparams[1]} LR {hparams[2]} Average correlation {np.average(corr)} Average precision {np.average(precision)} Average recall {np.average(recall)} Average F1-Score {np.average(f1_score)}"
            )

            return (
                np.average(corr),
                np.average(precision),
                np.average(recall),
                np.average(f1_score),
            )

    def run_one_experiment(self):
        summary_writer = self.__get_summary_writer()

        for train_dataset, validate_dataset in self.__k_fold_dataset:
            if not self.validate_only:
                self.train(train_dataset)

            avg_corr, avg_precision, avg_recall, avg_f1 = self.validate(
                validate_dataset
            )

            hparams = self.__get_current_experiment()
            model_name = self.__get_current_model().model_name

            summary_writer.add_hparams(
                {
                    "model": model_name,
                    "epoch": hparams[0],
                    "batch_size": hparams[1],
                    "lr": hparams[2],
                },
                {
                    "hparam/avg_corr": avg_corr,
                    "hparam/avg_precision": avg_precision,
                    "hparam/avg_recall": avg_recall,
                    "hparam/avg_f1": avg_f1,
                },
                run_name=os.path.dirname(self.__get_summary_save_path()),
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
