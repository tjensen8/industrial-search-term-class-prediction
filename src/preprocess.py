"""
Contains the preprocessing scripts that perform processing associated with:

- Train Test Split
- Tokenization
- Model Prep

"""

from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, List, AnyStr
from tqdm import tqdm
import logging
from multiprocessing import Pool

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

logging.basicConfig(
    filename="/home/programming/dsa-g/preprocess.log", level=logging.INFO, filemode="w"
)


def split_data(
    x_data: pd.DataFrame, y_data: pd.DataFrame, **kwargs
) -> Tuple[pd.DateOffset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # splits the data according to parameters specified for sklearn
    return train_test_split(x_data, y_data, random_state=0, **kwargs)


class SearchProcessor:
    """Preprocessor the contains all of the preprocessing needed for search term dataset.
    Designed in a modular way that can be used beyond initial analysis.

    Things that I have noticed in the text data.

    - Duplicate Spaces
    - Random multiple non-letter text (i.e 3 or more)
    - need lowercase
    -

    According to the huggingface tokenizer library, training your own tokenizer works as follows:
    SOURCE: https://huggingface.co/docs/tokenizers/quicktour
    - Start with all the characters present in the training corpus as tokens.
    - Identify the most common pair of tokens and merge it into one token.
    - Repeat until the vocabulary (e.g., the number of tokens) has reached the size we want.

    Source describing that custom-made word embeddings are distinct from general-purpose word models.
    In addition, they can present more novel short terms.
    SOURCE: https://aclanthology.org/2020.coling-demos.6.pdf

    """

    def __init__(self, texts: List[AnyStr]) -> None:
        self.texts = self._qa_run(texts)
        self.n_texts = len(self.texts)
        logging.info(f"Processed {self.n_texts}.")

    def _quality_assurance(self, text: AnyStr) -> str:
        """Ensures that each text in all of the texts provided is formatted correctly.
        Or, at the very least, formatted as a string.

        Args:
            texts (List[AnyStr]): List of texts to be processed.

        Returns:
            str: The texts that have been checked by QA.
        """
        # makes sure that text is formatted in a desired format
        # done before preprocessing

        # convert to string if not string
        if not isinstance(text, str):
            text = str(text)

        return text

    def _qa_run(self, texts: List):
        with Pool() as pool:
            texts = pool.map(self._quality_assurance, texts)
        return texts


def encode_classes(classes: list) -> Tuple[dict]:
    """Assign each of the classes a integer that represents the class.
    Returns an encoder and a decoder.

    Args:
        classes (list): List of unique classes.

    Returns:
        tuple: encoder and decoder dictionaries
    """
    if type(classes) != type(list()):
        classes = list(classes)
    # ensure classes are unique
    classes = list(set(classes))

    encoding = {}
    decoding = {}
    for counter, cls in enumerate(classes):
        decoding[counter] = cls
        encoding[cls] = counter

    # make dictionary of encoded classes
    return encoding, decoding


def get_stopwords(language: str = "english") -> bool:
    """Downloads the stopwords from NLTK to the local device. Returns the stopwords for language specified.

    Args:
        language (str): Language to return the stopwords for. Defaults to 'english',
    Returns:
        Bool: If download is successful or file already exists.
    """
    nltk.download("stopwords")

    return stopwords.words(language)


class SearchAnalyzer:
    """
    The analysis pipeline for different count vectors.
    """

    def __init__(self, corpus: List[AnyStr], stopwords: List[AnyStr] = None) -> None:
        self.corpus = corpus
        self.stopwords = stopwords

        self.word_counts = pd.DataFrame()
        self.summary_count = pd.DataFrame()

    def get_word_occur(self, ngram_range=(1, 1), **kwargs) -> pd.DataFrame:
        """Returns the count of how many times a word occured in the text.
        This analysis does not include any stopwords that have been provided.

        Args:
            corpus (list): List of strings that form documents in a corpus.
            stop_words (list, optional): List of stopwords to ignore. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe of the word(s) and the number of times the words appear in the corpus.
        """
        logging.info("Getting word occurance.")
        vec = CountVectorizer(
            stop_words=self.stopwords, ngram_range=ngram_range, **kwargs
        ).fit(self.corpus)
        bag_of_words = vec.transform(self.corpus)
        sum_words = bag_of_words.sum(axis=0)
        word_counts = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)
        word_counts = pd.DataFrame(word_counts, columns=["word", "count"])

        self.word_counts = word_counts
        logging.info("Complete.")

    def bin_counts(
        self, bin_intervals: List[Tuple[int]] = None, labels: List[AnyStr] = None
    ) -> pd.DataFrame:
        """Bin the counts of words from the corpus into usable and analyzable buckets. Done in memory.

        Args:
            bin_intervals (List[Tuple[int]], optional): The intervals to use. If no intervals are provided, intervals from 0-10 will be used. Defaults to None.
            labels (List[AnyStr], optional): The labels to sue for each of the bin intervals. This will be the final name of the bins. Defaults to None.

        Returns:
            pd.DataFrame: In-memory dataframe with the "count_bucket" column added.
        """
        assert (
            len(self.word_counts) != 0
        ), "Did you run get_word_occur() before this function?"
        logging.info("Binning the word counts together.")

        if bin_intervals == None:
            bin_intervals = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 11),
                (11, len(self.word_counts)),
            ]
        bins = pd.IntervalIndex.from_tuples(bin_intervals)

        if labels == None:
            labels = ["1", "2", "3", "4", "5", "5-10", "10+"]

        # add column to word counts dataframe
        self.word_counts["count_bucket"] = pd.cut(
            self.word_counts["count"], bins, labels=labels, right=True
        )

        # replace the word counts with the names of the categories
        self.word_counts["count_bucket"].cat.rename_categories(labels, inplace=True)
        logging.info("Complete.")

    def visualize(self, return_df=True):
        assert (
            len(self.word_counts) != 0
        ), "Did you run get_word_occur() and bin_counts() before this function?"
        logging.info("Visualizing summary word counts.")
        self.summary_count = self.word_counts.groupby("count_bucket").agg(
            bucket_count=("count_bucket", "count")
        )

        self.summary_count.plot(kind="bar", legend=False)
        plt.xlabel("Bucket of Word Counts")
        plt.ylabel("Number of Unique Words in the Bucket")
        plt.title("Number of Words By Their Occurrence")
        plt.show()

        self.summary_count.cumsum().plot(kind="line", legend=False)
        plt.xlabel("Cumulative Sume of Bucket of Word Counts")
        plt.ylabel("Number of Unique Words in the Bucket")
        plt.title("Number of Words By Their Occurrence")
        plt.show()

        vocab_size = self.summary_count["bucket_count"].sum()
        print(f"Total Vocab Size: {vocab_size}")

        self.summary_count[" Pct of Vocab Covered"] = round(
            (self.summary_count["bucket_count"] / vocab_size) * 100
        )
        logging.info("Complete.")
        if return_df:
            return self.summary_count


def set_count_vectorizer(
    stop_words: list, min_df: int, ngram_range: tuple, **kwargs
) -> CountVectorizer:
    """Creates a count vectorizer with the desired settings. Settings that are required are listed as arguments to this function. Other settings that are optional can be includeda as keyword arugments.

    Args:
        stop_words (list): List of stop words to leave out of the count vectorizer.
        min_df (int): Minimum document frequency. The minimum number of times that a word occurs in a search.
        ngram_range (tuple): Range of ngrams to include.

    Returns:
        CountVectorizer: _description_
    """
    return CountVectorizer(
        stop_words=stop_words, min_df=min_df, ngram_range=ngram_range, **kwargs
    )
