from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import wandb
import os

from dataset.pair_training_dataset import TrainingProductPairDataset
from helper.debug import IS_DEBUG
from helper.gpu import GPU_DEVICE
from models.pair_embedding import PairEmbeddingModel


class ModelTraining:
    def __init__(
        self,
        model: PairEmbeddingModel,
        model_epoch: int,
        model_lr: float,
        dataset: TrainingProductPairDataset,
        dataset_batch_size: int,
        summary_writer: SummaryWriter,
        model_save: bool = True,
        model_save_path: str = "./",
        model_loss=nn.CosineEmbeddingLoss(),
        random_seed: int = 69,
        dry_run: bool = False,
        gpu: bool = False,
    ) -> None:
        self.model = model
        self.model_epoch = model_epoch
        self.model_lr = model_lr
        self.dataset = dataset
        self.dataset_batch_size = dataset_batch_size
        self.model_save = model_save
        self.model_loss = model_loss
        self.summary_writer = summary_writer
        self.dry_run = dry_run
        self.gpu = gpu

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr
        )
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        now = datetime.now()
        self.model_save_path = f"{model_save_path}/pair_embedding_{model.model_name}_{model_epoch}_{model_loss}_{model_lr}_{now.year}_{now.month}_{now.day}_{now.hour}.pt"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)

        torch.manual_seed(random_seed)

    def load_model(self, path: str):
        self.model = torch.load(path)

    def get_data_loader(self, data):
        return DataLoader(
            data, batch_size=self.dataset_batch_size, shuffle=True
        )

    def train_each_epoch(self, epoch: int):
        self.model.train()
        self.model.model_train()

        training_data = self.get_data_loader(self.dataset)

        total_loss = 0.0

        for batch_idx, ((descriptions_1, description_2), labels) in tqdm(
            enumerate(training_data),
            desc="Training Batch",
            total=len(training_data),
        ):
            temp_label = labels
            if self.gpu:
                temp_label = torch.tensor(labels).to(GPU_DEVICE)

            self.optimizer.zero_grad()

            autocast_type = (
                torch.bfloat16
                if os.getenv("AUTOCAST_TYPE") == "BFLOAT"
                else torch.float16
            )

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    with torch.autocast(
                        "cuda" if self.gpu else "cpu", dtype=autocast_type
                    ):
                        model_outputs = self.model(
                            descriptions_1, description_2
                        )
                        loss = self.model_loss(*model_outputs, temp_label)

            if IS_DEBUG:
                print(
                    prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=25
                    )
                )
                print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=15))
                prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item() * len(descriptions_1)

            if IS_DEBUG:
                print("Model output", model_outputs)
                print("Model output shape", model_outputs[0].shape)
                print("Loss value", loss)
                print("Total loss", total_loss)

            if batch_idx % 100 == 0:
                print(f"\n\nLogging at {batch_idx}")

                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tRunning Loss: {:.6f}\n\n".format(
                        epoch,
                        batch_idx * len(descriptions_1),
                        len(training_data.dataset),
                        100.0 * batch_idx / len(training_data),
                        total_loss / ((batch_idx + 1) * len(descriptions_1)),
                    )
                )

                wandb.log(
                    {
                        "train/running_loss": total_loss
                        / ((batch_idx + 1) * len(descriptions_1))
                    }
                )

                if self.dry_run:
                    print("\n\nDRY RUN, BREAKING AFTER 100 BATCH SIZE\n\n")
                    break

            if (
                os.getenv("CLEAN_EVERY")
                and batch_idx % int(os.getenv("CLEAN_EVERY"))
                and self.gpu
                and self.dataset_batch_size > 8
            ):
                torch.cuda.memory.empty_cache()

        training_loss = total_loss / len(training_data.sampler)
        self.summary_writer.add_scalar(
            "Loss/Train", training_loss, global_step=epoch
        )
        wandb.log({"train/loss": training_loss})

    def train(self):
        with torch.enable_grad():
            for epoch in trange(1, self.model_epoch + 1, desc="Epoch"):
                self.train_each_epoch(epoch)

                print(f"\n\nDone with epoch {epoch}\n\n")

                self.model.clean_cache(self.gpu)

        if self.model_save:
            torch.save(self.model.state_dict(), self.model_save_path)
