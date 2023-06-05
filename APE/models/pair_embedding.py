from tempfile import mkdtemp

import os.path as path
import torch.nn as nn
import torch
import numpy as np
import random
import string
import gc


class PairEmbeddingModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super(PairEmbeddingModel, self).__init__()

        self.model_name = model_name

    def run_to_model_once(
        self, sentence_input: tuple[str, ...]
    ) -> torch.Tensor:
        return NotImplementedError("Please implement run_to_model_once method")

    def linear_feed_forward(
        self, pooling_output: torch.Tensor
    ) -> torch.Tensor:
        return NotImplementedError()

    def generate_temp_file_path(self):
        return path.join(
            mkdtemp(),
            "".join(
                random.choices(string.ascii_uppercase + string.digits, k=10)
            )
            + ".dat",
        )

    def to_memmap(self, array: np.ndarray):
        mmp = np.memmap(
            self.generate_temp_file_path(),
            dtype=array.dtype,
            mode="w+",
            shape=array.shape,
        )
        mmp[:] = array[:]
        mmp.flush()

        return mmp

    def clean_cache(self, gpu: bool):
        gc.collect()

        if gpu:
            torch.cuda.memory.empty_cache()

    def model_train(self):
        pass

    def model_eval(self):
        pass

    def model_reset(self):
        pass

    def forward(
        self, description_1: tuple[str, ...], description_2: tuple[str, ...]
    ):
        desc_1_emb = self.run_to_model_once(description_1)
        desc_2_emb = self.run_to_model_once(description_2)

        desc_min = torch.abs(torch.sub(desc_1_emb, desc_2_emb))
        desc_cat = torch.cat([desc_1_emb, desc_2_emb, desc_min], dim=1)

        ffnn = self.linear_feed_forward(desc_cat)

        return ffnn
