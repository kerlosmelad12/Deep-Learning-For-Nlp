from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
lemtizer=WordNetLemmatizer()

nltk.download('wordnet')
stem=PorterStemmer()
snowball=SnowballStemmer('english')
text= "115712 I understand I would like to assist you We would need to get you into a private secured link to further assist"

def portsteming(text):
    '''
    Normalize text by Stemming for english
   
    
    parameters:
    Text (str):input text
    Returns:
    Normalized Text by removing suffiex or prefixes
    '''
    stem_text=[]
    for word in str(text).split():
        stem_text.append(stem.stem(word))
    return ' '.join(stem_text)

def snowballstemmerr(text):
    '''
    Normalize text by Stemming for any languages including english
   
    
    parameters:
    Text (str):input text
    Returns:
    Normalized Text by removing suffiex or prefixes
    '''
   
    snowballstem=[]
    for word in str(text).split():
        snowballstem.append(snowball.stem(word))
    return ' '.join(snowballstem)

def lemitization(text):
    '''
    Normalize text by lemitizing based on the languties or in grammer posation 
    
    parameters:
    Text (str):input text
    Returns:
    Normalized Text '''
    lemitizedtext=[]
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    postag=nltk.pos_tag(str(text).split())
    for w,pos in postag:
        pos=wordnet_map.get(pos[0],wordnet.NOUN)
        lemitizeword=lemtizer.lemmatize(w,pos=pos)
        lemitizedtext.append(lemitizeword)

    return ' '.join(lemitizedtext)





print("Original Text:")
print(text)
print("-" * 50)

print("Porter Stemming:")
print(portsteming(text))
print("-" * 50)

print("Snowball Stemming:")
print(snowballstemmerr(text))
print("-" * 50)

print("Lemmatization:")
print(lemitization(text))
print("-" * 50)









