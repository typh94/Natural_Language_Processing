import ast
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

questions = []
answers = []

with open('qa_Video_Games.json', 'r') as f:
    for line in f:
        data = ast.literal_eval(line)
        questions.append(data['question'].lower())
        answers.append(data['answer'].lower())

print('Data on questions:', len(questions))

for index, val in enumerate(questions[:20]):
    print('[', index, ']', val)

# use CountVectorizer to convert the questions list into a sparse matrix
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(questions)

tfidf = TfidfTransformer(norm='l2')
X_tfidf = tfidf.fit_transform(X_vec)


def conversation(user_question):
    global tfidf, answers, X_tfidf
    # Vectorize user's question
    Y_vec = vectorizer.transform([user_question])
    Y_tfidf = tfidf.transform(Y_vec)

    angle = np.rad2deg(np.arccos(max(cosine_similarity(Y_tfidf, X_tfidf)[0])))
    if angle > 60:
        return 'Sorry, I did not quite understand that.'
    else:
        return answers[np.argmax(cosine_similarity(Y_tfidf, X_tfidf)[0])]


user = input('Please enter your name:')
print('Hi', user, 'I am a rep of Q&A support. How can I help you?')

while True:
    user_question = input("{}: ".format(user))

    if user_question.lower() == 'bye':
        print('Q&A support: Bye! Have a good day!')
        break
    else:
        print('The question is:', user_question)
        print('Q&A support:', conversation(user_question))




##Questions working
        # Would it work if I submerge the game in water ?
        # What are your political views ?
        # What's your biggest passion ?
        # Are you scared of humans ?
        # Is the game as good as it was said to be ?
##Questions not working
        # Would it work if the game gets wet
        # Do you have any hobbies ?
        # what do you think of biases being implemented in AI ?
        # What is your favorite software ?
        # Do you like it when users ask you questions ?

        
