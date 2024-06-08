#Build a basic language model using trigrams of the Reuter corpus
#Reuteurs corpus is a collection of 10,788 news documents
#totalling 1.3 million words in the corpus

import nltk
nltk.download('reuters')
from nltk.corpus import reuters
#print(type(reuters))
sentences = reuters.sents()
print(sentences[:10])
print('There are', len(sentences),'sentences.')

#Text cleaning- remove punctuation and fix contractions
#pip install contractions. But contractions only work on strings, not list of words/. 
#here we have token thats why it does not work
import contractions
import string

cleaned_up1 = []
for s in sentences:
    text_cleaned = []
    for m in s:
        if m!= '.' and m!= '""' and m!= '>' and m!=',' and m!= '-':
            text_cleaned.append(m)
    cleaned_up1.append(text_cleaned)
print(cleaned_up1[:10])
print()
print()

#lowercase the words
cleaned_up = []
for s in cleaned_up1:
    new_sent = [''.join(char for char in item
                        if char not in string.punctuation) #Not working?
                for item in s if item !='']
    text_cleaned=[]
    for x in new_sent:
        if x!= '':
            text_cleaned.append(x.lower())
    cleaned_up.append(text_cleaned)

print('*****************CLEANED UP******************')
print(cleaned_up[:10])

#N-Grams
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

#Create a placeholder for language model
model = defaultdict(lambda: defaultdict(lambda:0))
#Split each sentence into trigrams and then calculate the frequency
#in which each combination of trigrams occurs in the reuters data

for sentence in cleaned_up:
    for word1, word2, word3 in trigrams(sentence, pad_right=True,pad_left=True):
        model[(word1,word2)] [word3] +=1
        #Word1, word2 as predictor of word3
#Use the frequency to calculate probabilities of a word given the previous two words
for word1_word2 in model:
    total_count=float(sum(model[word1_word2].values()))
    for word3 in model[word1_word2]:
        model[word1_word2][word3] = model[word1_word2][word3] / total_count

#Now we have the language model, can be used to make predictions of "next" words
#For example:
print(dict(model["buisnessmen","and"]))
print()
predictors = dict(model["buisnessmen","and"])
for x,y in predictors.items():
    print(x, '\t',y)
print()
print('*******************NEXT WORD*******************')
max_next_word = max(predictors.items(),key=lambda a:a[1])









        






























        
