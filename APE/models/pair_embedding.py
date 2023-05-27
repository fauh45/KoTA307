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
        super().__init__()

        self.model_name = model_name

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

    def forward(self, anc_desc: str, pos_desc: str, neg_desc: str):
        anc = self.run_to_model_once(anc_desc)
        pos = self.run_to_model_once(pos_desc)
        neg = self.run_to_model_once(neg_desc)

        return anc, pos, neg
