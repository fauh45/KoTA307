from tempfile import mkdtemp

import os.path as path
import torch.nn as nn
import numpy as np
import random
import string


class PairEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, gpu: bool) -> None:
        super().__init__()

        self.model_name = model_name
        self.gpu = gpu

    def run_to_model_once(self, sentence_input: str):
        return NotImplementedError("Please implement run_to_model_once method")

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

    def model_train(self):
        pass

    def model_eval(self):
        pass

    def model_reset(self):
        pass

    def forward(self, description_1: str, description_2: str):
        desc_1_emb = self.run_to_model_once(description_1)
        desc_2_emb = self.run_to_model_once(description_2)

        return desc_1_emb, desc_2_emb
