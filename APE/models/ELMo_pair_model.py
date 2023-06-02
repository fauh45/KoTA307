import torch

from elmoformanylangs import Embedder
from torch.autograd import Variable

from models.pair_embedding import PairEmbeddingModel


class ELMoPairModel(PairEmbeddingModel):
    def __init__(self, pretrained_model_dir: str, gpu: bool) -> None:
        super().__init__("ELMo")

        self.__pretrained_model_dir = pretrained_model_dir

        self.ffnn = torch.nn.Linear(1024 * 3, 3)

        self.__embedder = Embedder(pretrained_model_dir)
        self.__model = self.__embedder.model
        self.gpu = gpu

        if gpu:
            self.ffnn = self.ffnn.to("cuda")

    def __split_description(self, description: str):
        cleaned_desc = "".join(filter(str.isalnum, description))
        cleaned_desc = cleaned_desc.split(" ")

        return [cleaned_desc]

    def run_to_model_once(self, sentence_input: str):
        # Updated the sents2elmo using tensor instead, but still need to make sure
        tensored = self.__embedder.sents2elmo(
            self.__split_description(sentence_input)
        )[0]

        if not self.gpu:
            tensored = tensored.cpu()

        return tensored

    def linear_feed_forward(
        self, pooling_output: torch.Tensor
    ) -> torch.Tensor:
        return self.ffnn(pooling_output)

    def model_eval(self):
        self.__model.eval()

    def model_train(self):
        self.__model.train()

    def model_reset(self):
        del self.__embedder

        self.clean_cache(self.gpu)

        self.__embedder = Embedder(self.__pretrained_model_dir)
        self.__model = self.__embedder.model
