# Simple illustration of CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
data = ["aa bb cc", "cc dd ee", "bb dd cc"]
print('data', data)
print()
mycountvectorizer = CountVectorizer()

# Breaking .fit_transform up into .fit and .transform
vector = mycountvectorizer.fit(data)
print(vector.get_feature_names_out())

# Encode the document:
vector = mycountvectorizer.transform(data)
print()
print(vector.toarray())

# Doing the combination:
vectorcombo = mycountvectorizer.fit_transform(data)
print('\n\n And when combining the two methods:')
print(vectorcombo.toarray())


#====================================
# Now for the normalization -- we use this output as input
# to the TfidfTransformer method:

from sklearn.feature_extraction.text import TfidfTransformer
mytfidft = TfidfTransformer(norm='l2')

# use l2 for cosine similarity application later

X_tfidf = mytfidft.fit_transform(vectorcombo)

print('\n\n\n Tfidf Tranformer normalization: ')

def printA(a):
    for row in a:
        for col in row:
            print("{:8.3f}".format(col), end=" ")
        print("")
    
printA(X_tfidf.toarray())
