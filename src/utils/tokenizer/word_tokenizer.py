from abc import ABC, abstractmethod

from nltk.tokenize import word_tokenize


class WordTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError("Subclasses must implement this method")


class NLTKWordTokenizer(WordTokenizer):
    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(text)


class BasicWordTokenizer(WordTokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split()
