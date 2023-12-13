import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence:str):
    """
    Tokenize a given sentence into an array of words/tokens.

    Parameters:
    - sentence (str): The input sentence to be tokenized.

    Returns:
    - tokens (list): An array of words/tokens extracted from the input sentence.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Perform stemming on a given word to find its root form.

    Stemming is the process of reducing a word to its base or root form.
    For example:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]

    Parameters:
    - word (str): The word to be stemmed.

    Returns:
    - stemmed_word (str): The root form of the input word after stemming.
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    Create a bag-of-words representation for a given tokenized sentence and a set of predefined words.

    A bag-of-words representation is a vector that counts the frequency of each word in the given sentence
    based on the predefined set of words.

    Parameters:
    - tokenized_sentence (list): A list of words/tokens representing a sentence.
    - words (list): A predefined list of unique words used for creating the bag-of-words representation.

    Returns:
    - bag (list): A binary vector indicating the presence (1) or absence (0) of each word in the predefined set.
    """
    pass


