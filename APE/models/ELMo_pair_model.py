from elmoformanylangs import Embedder
import torch
import numpy as np

from models.pair_embedding import PairEmbeddingModel


class ELMoPairModel(PairEmbeddingModel):
    def __init__(self, pretrained_model_dir: str, gpu: bool) -> None:
        super().__init__("ELMo")

        self.__pretrained_model_dir = pretrained_model_dir

        self.__embedder = Embedder(pretrained_model_dir)
        self.__model = self.__embedder.model
        self.gpu = gpu

    def __split_description(self, description: str):
        cleaned_desc = "".join(filter(str.isalnum, description))
        cleaned_desc = cleaned_desc.split(" ")

        return [cleaned_desc]

    def run_to_model_once(self, sentence_input: str):
        # TODO: Not really sure about this one, is it really updating the weights of the ELMo model?
        tensored = torch.tensor(
            self.__embedder.sents2elmo(
                self.__split_description(sentence_input)
            ),
            requires_grad=True,
        )[0]

        if self.gpu:
            tensored = tensored.to("cuda")

        return tensored

    def model_eval(self):
        self.__model.eval()

    def model_train(self):
        self.__model.train()

    def model_reset(self):
        self.__embedder = Embedder(self.__pretrained_model_dir)
        self.__model = self.__embedder.model
