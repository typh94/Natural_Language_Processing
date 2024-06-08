#DONT FORGET TO EXTRACT THE ZIP FILE
import nltk
from nltk import word_tokenize

#Import data from the file
with open('UFOReportsWIspring23.txt', 'r', encoding= 'utf-8') as file:
    UFO = file.read()

print(UFO)

#sentence:
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(UFO)

print(sentences)
for index,s in enumerate(sentences):
    print(index, "  ",s)

from nltk.tokenize import word_tokenize
#We need a string to tokenize. here we have a list of strings, so need to flatten
bigstrings = ''
for s in sentences:
    bigstrings = bigstrings + s #Concatenate each sentence into one big sentence
    
#Replace '.' with ' '
bigstrings = bigstrings.replace('.',' ')


print(bigstrings)
word_tokens = word_tokenize(bigstrings)
print("******WORDS******")
print(word_tokens)

#****************STEMMING*************Porter and Snowball 
from nltk.stem import PorterStemmer 
from nltk.stem import SnowballStemmer

ss = SnowballStemmer(language='english')
ps = PorterStemmer()

#CAN remove stopwords
#Remove stopwords from the list of tokens 
##import nltk
##nltk.download('stopwords')
##stopwords = nltk.corpus.stopwords.words('english')

##for w in word_tokens:
##    print(w,"Porter Stem==>", ps.stem(w), "Snowball Stem==>", ss.stem(w))
##    print()

#Texblob lemmetize
from textblob import TextBlob, Word #textblob lemmetizes sentences not words.
sent2 = [] #LIST of lemmetized sentences
for s in sentences: #using one sentence at a time
    blob_sent= TextBlob(s)
    sent2.append(s)

lemm_list = []
for one_blob_sentence in sent2:
    lem_sentence = "".join([w.lemmetize() for w in one_blob_sentence])
    lemm_list.append(lem_sentence)
print(lemm_list)
    
    
 




