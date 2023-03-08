import torch.nn as nn


class PairEmbeddingModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name

    def run_to_model_once(self, sentence_input: str):
        return NotImplementedError("Please implement run_to_model_once method")

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
