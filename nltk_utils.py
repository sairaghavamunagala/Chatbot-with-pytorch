import nltk
import numpy as np
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
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


