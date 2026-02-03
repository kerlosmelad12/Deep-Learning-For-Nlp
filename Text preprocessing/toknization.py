from nltk.tokenize import word_tokenize,sent_tokenize,TreebankWordTokenizer,WordPunctTokenizer
import nltk

text = "Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence. It enables machines to understand human language, analyze text, and extract meaningful insights. For example, NLP is used in spam detection, chatbots, search engines, and sentiment analysis. Isn't it amazing how computers can understand language?"


def sentence_tokenization(text):
    '''
    split the text based on statical role to segment a text into a list of sentence
    The sent_tokenize function uses an instance of PunktSentenceTokenizer from 
    the nltk.tokenize.punkt module, which is already been trained and thus very well knows
     to mark the end and beginning of sentence at what characters and punctuation.
     it load from r'tokenizers/punkt/PY3/english.pickle'
    '''
    tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
    sentence=tokenizer.tokenize(text)
    return sentence



def wordtokenizer(text):
    '''
      The word_tokenize function is helpful for breaking down a sentence or text into its constituent words.
      Eases analysis or processing at the word level in natural language processing tasks.  
    '''
    return word_tokenize(text)

def TreebankWordTokenizerr(text):
    '''
    These tokenizers work by separating the words using punctuation and spaces.
      allowing a user to decide what to do with the punctuations at the time of pre-processing.
    '''
    tokenizer=TreebankWordTokenizer()
    return tokenizer.tokenize(text)


def wordpuncttoknizer(text):
    '''
    The WordPunctTokenizer is one of the NLTK tokenizers that splits words based on punctuation boundaries.
    Each punctuation mark is treated as a separate token.
    '''
    tokenizer=WordPunctTokenizer()
    return tokenizer.tokenize(text)



print("Sentence Tokenization:")
print(sentence_tokenization(text))

print("\nWord Tokenization (word_tokenize):")
print(wordtokenizer(text))

print("\nTreebank Word Tokenizer:")
print(TreebankWordTokenizerr(text))

print("\nWord Punctuation Tokenizer:")
print(wordpuncttoknizer(text))