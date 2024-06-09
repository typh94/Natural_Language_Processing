                         ###Netflix Show descriptions###

#To load the text dataset
import matplotlib.pyplot as plt
with open('netflixshowdescriptions.txt', 'r', encoding='utf-8') as file:
    netflix = file.readlines()
  
    
#a. Print first 10 lines
print()
print('First 10 lines from dataset:')
print(netflix[:10])


#b. Print the total of lines in the dataset
print()
print('How many lines do we have?', len(netflix))


#c. Strip away newline characters
netflix = list(filter(None,[item.strip('\n') for item in netflix]))


#d. Print count of the number of non-blank lines in dataset
print()
print('Removing blank lines:')
print(netflix[:10])

print('How many lines do we have?')
print(len(netflix))


#e. Graph the numbers of characters per sentence using a histogram plot
line_lengths =[len(sentence) for sentence in netflix]
plt.hist(line_lengths)
plt.title("number of characters per sentence")
plt.show()


#f. Break sentences into token
tokens = [item.split() for item in netflix]


#g. Print tokens resulting from first 5 lines in dataset
print()
print("Tokenization:")
print (tokens [:5])


#h. Print  number of tokens per sentence for first 10 lines of  dataset
tokens_per_sentence = [len(sentence.split()) for sentence in netflix]
print()
print(tokens_per_sentence[:10])


#i. Graph the number of tokens per senrence in a histogram plot
plt.hist(tokens_per_sentence, color='teal')
plt.title('Number of words per sentence')
plt.show()


#j. Flatten list of tokens into one big list of tokens
words = [word for sentence in tokens for word in sentence]
print()
print('First 140 words:')
print(words[:140])
print('Number of words in the Netflic Show Description:', len(words))


#k. Print the first 20 tokens in the new list
print()
print('First 20 tokens in the list:')
print(words[:20])


#l. Print the 10 most common words in the text corpus
##lowercase all of the words in the text
words = [word.lower() for word in words]
print()
print(words)
from collections import Counter
c = Counter(words)
c10 = c.most_common(10)
print()
print('10 most common words:')
for x in c10:
    print(x)


#m. Remove stopwords from the list of tokens 
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


#n. Print 10 most common non-stop words in text corpus
words = [word.lower() for word in words if word.lower() not in stopwords]
##Display 10 most popular stopwords
from collections import Counter
c = Counter(words)
c10 = c.most_common(10)
print()
print('The 10 most common words in this Netflix Show Description are:')
for x in c10:
    print(x)


#o. Create a WordCloud with a bilinear interpolation with axis off and plot it
from wordcloud import WordCloud
words_string = " ".join(words)
my_wordcloud = WordCloud().generate(words_string)

##Display the generated image
plt.imshow(my_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=30,
                      background_color="white"). generate (words_string)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


#p. Create a 2nd WordCloud that plots a maximum of 40 words, on red background
from wordcloud import WordCloud
words_string = " ".join(words)
my_wordcloud = WordCloud().generate(words_string)

##Display the generated image
plt.imshow(my_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()
wordcloud = WordCloud(max_words=40, background_color='red',
                      contour_width=5, contour_color='green').generate(words_string)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()



#q. Create a 3rd WordCloud that plots a maximum of 40 words using image mask on Canvas

##pip install pillow
from PIL import Image
import numpy as np
from wordcloud import WordCloud

#generate function needs a string of words so
words_string = " ".join(words)
word_mask = np.array(Image.open ('my_wordcloud.png'))
wordcloud = WordCloud(max_words=40, background_color='white', mask=word_mask,
                      contour_width=5, contour_color='green').generate(words_string)
#Display the WordCloud
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()


























    
