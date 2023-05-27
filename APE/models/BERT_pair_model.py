from transformers import AutoTokenizer, AutoModel

from models.pair_embedding import PairEmbeddingModel


class BERTPairModel(PairEmbeddingModel):
    def __init__(self, pretrained_name: str, gpu: bool) -> None:
        super().__init__("BERT")

        self.__pretrained_name = pretrained_name

        self.__tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.__model = AutoModel.from_pretrained(pretrained_name)

        self.gpu = gpu
        if gpu:
            self.__model = self.__model.to("cuda")

    def __tokenize(self, sentence_input: str):
        tokenizer_result = self.__tokenizer(
            sentence_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        if self.gpu:
            tokenizer_result = tokenizer_result.to("cuda")

        return tokenizer_result

    def run_to_model_once(self, sentence_input: str):
        tokenized = self.__tokenize(sentence_input)
        output = self.__model(**tokenized)

        return output.pooler_output

    def model_eval(self):
        self.__model.eval()

    def model_train(self):
        self.__model.train()

    def model_reset(self):
        del self.__tokenizer
        del self.__model

        self.clean_cache(self.gpu)

        self.__tokenizer = AutoTokenizer.from_pretrained(
            self.__pretrained_name
        )
        self.__model = AutoModel.from_pretrained(self.__pretrained_name)

        if self.gpu:
            self.__model = self.__model.to("cuda")
