import torch

from elmoformanylangs import Embedder

from models.pair_embedding import PairEmbeddingModel


class ELMoPairModel(PairEmbeddingModel):
    def __init__(self, pretrained_model_dir: str, gpu: bool) -> None:
        super().__init__("ELMo")

        self.__pretrained_model_dir = pretrained_model_dir

        self.__embedder = Embedder(pretrained_model_dir)
        self.__model = self.__embedder.model
        self.gpu = gpu

    def __split_description(self, description: tuple[str, ...]):
        cleaned_desc = []

        temp_description = description
        if isinstance(description, str):
            temp_description = [description]

        for desc in temp_description:
            temp = desc.split(" ")

            cleaned_desc.append(temp)

        return cleaned_desc

    def run_to_model_once(self, sentence_input: tuple[str, ...]):
        # Updated the sents2elmo using tensor instead, but still need to make sure
        tensored = self.__embedder.sents2elmo(
            self.__split_description(sentence_input)
        )
        tensored = torch.stack(tensored, dim=0)

        if not self.gpu:
            tensored = tensored.cpu()

        return tensored

    def model_eval(self):
        self.__model.eval()

    def model_train(self):
        self.__model.train()

    def model_reset(self):
        del self.__embedder

        self.clean_cache(self.gpu)

        self.__embedder = Embedder(self.__pretrained_model_dir)
        self.__model = self.__embedder.model
