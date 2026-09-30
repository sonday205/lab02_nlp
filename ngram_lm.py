"""Small, from-scratch unigram, bigram, and trigram language models."""

from collections import Counter
import math
import re
from typing import Iterable, Sequence


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep words/numbers as tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def build_vocabulary(corpus: Iterable[Sequence[str]]) -> set[str]:
    return {word for sentence in corpus for word in sentence}


def count_ngrams(corpus: Iterable[Sequence[str]], n: int) -> Counter:
    if n < 1:
        raise ValueError("n must be at least 1")
    counts = Counter()
    for sentence in corpus:
        for index in range(len(sentence) - n + 1):
            counts[tuple(sentence[index:index + n])] += 1
    return counts


class NGramLanguageModel:
    """MLE n-gram model with optional add-one conditional smoothing."""

    def __init__(self, n: int, smoothing: str = "mle"):
        if n not in (1, 2, 3):
            raise ValueError("Only unigram, bigram, and trigram models are supported")
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing must be 'mle' or 'laplace'")
        self.n = n
        self.smoothing = smoothing
        self.vocabulary: set[str] = set()
        self.ngram_counts: Counter = Counter()
        self.unigram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.total_tokens = 0

    def fit(self, corpus: Iterable[Sequence[str]]):
        sentences = [tuple(sentence) for sentence in corpus if sentence]
        self.vocabulary = build_vocabulary(sentences)
        self.ngram_counts = count_ngrams(sentences, self.n)
        self.unigram_counts = count_ngrams(sentences, 1)
        self.total_tokens = sum(len(sentence) for sentence in sentences)
        if self.n > 1:
            self.context_counts = count_ngrams(sentences, self.n - 1)
        return self

    def probability(self, context: Sequence[str], word: str) -> float:
        word = word.lower()
        if self.n == 1:
            count = self.ngram_counts.get((word,), 0)
            denominator = self.total_tokens
            vocabulary_size = max(1, len(self.vocabulary))
        else:
            context = tuple(token.lower() for token in context[-(self.n - 1):])
            if len(context) != self.n - 1:
                return 0.0
            count = self.ngram_counts.get(context + (word,), 0)
            denominator = self.context_counts.get(context, 0)
            vocabulary_size = max(1, len(self.vocabulary))
        if self.smoothing == "laplace":
            return (count + 1) / (denominator + vocabulary_size)
        return count / denominator if denominator else 0.0

    def _tokens(self, sentence: str | Sequence[str]) -> list[str]:
        words = tokenize(sentence) if isinstance(sentence, str) else [w.lower() for w in sentence]
        # A single unknown token keeps held-out probability finite with Laplace smoothing.
        return [word if word in self.vocabulary else "<unk>" for word in words]

    def sentence_log_probability(self, sentence: str | Sequence[str]) -> float:
        words = self._tokens(sentence)
        if not words:
            return 0.0
        log_probability = 0.0
        for index, word in enumerate(words):
            if self.n == 1:
                context = ()
            else:
                context = words[max(0, index - self.n + 1):index]
                if len(context) < self.n - 1:
                    # Initial tokens have no complete history; use the unigram distribution.
                    count = self.unigram_counts.get((word,), 0)
                    p = ((count + 1) / (self.total_tokens + max(1, len(self.vocabulary)))
                         if self.smoothing == "laplace" else count / self.total_tokens if self.total_tokens else 0.0)
                    if p == 0.0:
                        return -math.inf
                    log_probability += math.log(p)
                    continue
            p = self.probability(context, word)
            if p == 0.0:
                return -math.inf
            log_probability += math.log(p)
        return log_probability

    def sentence_probability(self, sentence: str | Sequence[str]) -> float:
        log_probability = self.sentence_log_probability(sentence)
        return math.exp(log_probability) if math.isfinite(log_probability) else 0.0

    def perplexity(self, corpus: Iterable[str | Sequence[str]]) -> float:
        total_log_probability = 0.0
        total_words = 0
        for sentence in corpus:
            words = self._tokens(sentence)
            if not words:
                continue
            total_words += len(words)
            score = self.sentence_log_probability(words)
            if not math.isfinite(score):
                return math.inf
            total_log_probability += score
        return math.exp(-total_log_probability / total_words) if total_words else math.nan

    def next_word_distribution(self, context: str | Sequence[str]) -> dict[str, float]:
        if isinstance(context, str):
            context_words = tokenize(context)
        else:
            context_words = [word.lower() for word in context]
        if self.n == 1:
            context_words = []
        else:
            context_words = [w if w in self.vocabulary else "<unk>" for w in context_words]
        candidates = sorted(self.vocabulary)
        if "<unk>" not in self.vocabulary:
            candidates.append("<unk>")
        return {word: self.probability(context_words, word) for word in candidates}

    def predict_next(self, context: str | Sequence[str], top_k: int = 5) -> list[tuple[str, float]]:
        distribution = self.next_word_distribution(context)
        return sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def train_unigram(corpus: Iterable[Sequence[str]], smoothing: str = "mle") -> NGramLanguageModel:
    return NGramLanguageModel(1, smoothing).fit(corpus)


def train_bigram(corpus: Iterable[Sequence[str]], smoothing: str = "mle") -> NGramLanguageModel:
    return NGramLanguageModel(2, smoothing).fit(corpus)


def train_trigram(corpus: Iterable[Sequence[str]], smoothing: str = "mle") -> NGramLanguageModel:
    return NGramLanguageModel(3, smoothing).fit(corpus)
