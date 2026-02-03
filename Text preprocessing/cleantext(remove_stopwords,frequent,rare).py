import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re

# --------------------------------------------------
# Make sure required NLTK resources are downloaded
# --------------------------------------------------
# nltk.download('punkt')
# nltk.download('stopwords')

# Load English stopwords
english_stopwords = set(stopwords.words('english'))


# --------------------------------------------------
# Function: remove_stopwords
# --------------------------------------------------
def remove_stopwords(text):
    """
    Remove English stopwords from a given text.

    Parameters:
    text (str): Input text

    Returns:
    str: Text after removing stopwords
    """
    clean_text = []

    for word in word_tokenize(text.lower()):
        if word.isalpha() and word not in english_stopwords:
            clean_text.append(word)

    return ' '.join(clean_text)


# --------------------------------------------------
# Function: remove_frequent_words
# --------------------------------------------------
def remove_frequent_words(text, top_n=10):
    """
    Remove the most frequent words from the text.

    Why remove frequent words?
    - They often carry little semantic meaning
    - Reduce noise in NLP tasks
    - Improve model performance

    Parameters:
    text (str): Input text
    top_n (int): Number of most frequent words to remove

    Returns:
    str: Text after removing frequent words
    """
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    count = Counter(tokens)

    # Get top N frequent words
    frequent_words = {word for word, _ in count.most_common(top_n)}

    clean_text = [word for word in tokens if word not in frequent_words]

    return ' '.join(clean_text)


# --------------------------------------------------
# Function: remove_rare_words
# --------------------------------------------------
def remove_rare_words(text, rare_n=10):
    """
    Remove rare words from the text.

    Why remove rare words?
    - Rare words may be typos or noise
    - Help stabilize ML models

    Parameters:
    text (str): Input text
    rare_n (int): Number of least frequent words to remove

    Returns:
    str: Text after removing rare words
    """
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    count = Counter(tokens)

    # Get least frequent words
    rare_words = {word for word, _ in count.most_common()[-rare_n:]}

    clean_text = [word for word in tokens if word not in rare_words]

    return ' '.join(clean_text)


# --------------------------------------------------
# Example Usage
# --------------------------------------------------
text = "115712 I understand I would like to assist you We would need to get you into a private secured link to further assist"

print("Original Text:\n", text)
print("\nAfter Stopword Removal:\n", remove_stopwords(text))
print("\nAfter Frequent Word Removal:\n", remove_frequent_words(text, top_n=5))
print("\nAfter Rare Word Removal:\n", remove_rare_words(text, rare_n=5))