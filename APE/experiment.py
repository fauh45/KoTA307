from itertools import product
from typing import List
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
from tqdm import tqdm

import os
import torch
import numpy as np
import wandb

from dataset.simple_split_dataset import SimpleSplitDataset
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
        train_val_split: float = 0.8,
        start_models_on: int = 0,
        start_experiment_on: int = 0,
        save_dir: str = "",
        dry_run: bool = False,
        validate_only: bool = False,
        gpu: bool = False,
        save_dataset: bool = False,
    ) -> None:
        self.__simple_split_dataset = SimpleSplitDataset(
            dataset_path,
            min_product_bought,
            train_val_split,
            save=save_dataset,
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
        self.gpu = gpu

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

    def __get_current_model(self):
        return self.models[self.current_models_index]

    def __create_embeddings(self, description: str):
        if self.gpu:
            return (
                self.__get_current_model()
                .run_to_model_once(description)
                .cpu()
                .numpy()
            )

        return (
            self.__get_current_model().run_to_model_once(description).numpy()
        )

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
            gpu=self.gpu,
            model_save=False,
        )

        training_model.train()

    def validate(self, dataset: RecommendationValidationDataset, n_rec: int):
        current_model = self.__get_current_model()

        current_model.eval()
        current_model.model_eval()

        hparams = self.__get_current_experiment()

        all_unique_item = self.__simple_split_dataset.unique_items

        with torch.no_grad():
            # * Done because modin cannot pickle "__create_embeddings"
            all_unique_item["embeddings"] = [
                self.__create_embeddings(desc)
                for desc in tqdm(
                    all_unique_item["Product description"].values,
                    "Generating embeddings for all unique items",
                )
            ]

            temp_unique_item = all_unique_item.reset_index()
            temp_unique_item["embeddings"] = temp_unique_item[
                "embeddings"
            ].apply(lambda x: x[0].tolist())

            wandb.log({"unique_items": temp_unique_item})

            def get_embeddings(sku: str):
                return all_unique_item.at[sku, "embeddings"]

            validation_data = DataLoader(
                dataset, batch_size=hparams[1], collate_fn=lambda x: x
            )

            corr = []
            precision = []
            recall = []
            f1_score = []

            final_data = []

            for batch in tqdm(validation_data, "Validating bath"):
                for seed, label in tqdm(
                    batch, "Validating each case in batch"
                ):
                    seed_embeddings = get_embeddings(seed["Lineitem sku"])

                    embeddings_distance = (
                        all_unique_item.loc[
                            all_unique_item.index != seed["Lineitem sku"]
                        ]["embeddings"]
                        .apply(
                            lambda emb2: cosine_distances(
                                seed_embeddings, emb2
                            )
                        )
                        .sort_values(ascending=True)
                        .head(n_rec)
                    )

                    selected_items = all_unique_item[
                        all_unique_item.index.isin(embeddings_distance.index)
                    ]

                    final_data.append(
                        [
                            seed.to_json(orient="records"),
                            label.to_json(orient="records"),
                            selected_items.to_json(orient="records"),
                        ]
                    )

                    n_recommended = n_rec
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
                        if i >= len(label): break

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

                wandb.log(
                    {
                        f"validate/corr-{n_rec}": np.average(corr),
                        f"validate/precision-{n_rec}": np.average(precision),
                        f"validate/recall-{n_rec}": np.average(recall),
                        f"validate/f1-{n_rec}": np.average(f1_score),
                    }
                )

                if self.dry_run:
                    print(
                        "\n\nEXITING VALIDATION AFTER ONE BATCH ON DRY RUN\n\n"
                    )
                    break

            wandb.log(
                {
                    f"validate_result_{n_rec}": wandb.Table(
                        columns=["seed", "label", "selected"],
                        data=final_data,
                    )
                }
            )

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
        (
            train_dataset,
            validate_dataset,
        ) = self.__simple_split_dataset.get_dataset()

        hparams = self.__get_current_experiment()
        model_name = self.__get_current_model().model_name

        wandb.init(
            project="APE-KoTa307",
            config={
                "dry_run": self.dry_run,
                "gpu": self.gpu,
                "model": model_name,
                "epoch": hparams[0],
                "batch_size": hparams[1],
                "lr": hparams[2],
            },
            reinit=True,
            name=f"TRIPLETS_{'DRY_RUN' if self.dry_run else 'FULL'}_"
            + self.__get_summary_comment()
            + f"_TIME_{datetime.now().isoformat()}",
        )

        if not self.validate_only:
            self.train(train_dataset)

        for n in tqdm([5, 10, 25, 50, 100]):
            self.validate(validate_dataset, n)

    def run_experiment(self):
        for _ in range(len(self.models)):
            print("\n\nRUNNING NEW MODEL")
            print("CURRENT MODEL", self.__get_current_model().model_name)
            print("\n\n")

            for _ in range(len(self.experiment_hparams)):
                self.run_one_experiment()

                print("\n\nMOVING EXPERIMENT FORWARD")
                self.__move_experiment_index_forward()
                print("CURRENT EXPERIMENT", self.__get_current_experiment())
                print("\n\n")

                self.__reset_current_model()

                if self.dry_run:
                    print("\n\nBREAKING AFTER ONE HPARAMS ON DRY RUN\n\n")
                    break

                if self.current_experiment_index > len(
                    self.experiment_hparams
                ):
                    break

            self.__move_model_index_forward()

            if self.current_models_index > len(self.models):
                break
