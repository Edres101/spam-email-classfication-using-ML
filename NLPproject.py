import nltk
import string
import pandas
import numpy as np 
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score



def print_classification_report(model_name, model):
    model.fit(training_data, training_labels)

    #Evaluate the model on the training data set
    prediction = model.predict(training_data)
    print("\nBest Model is:", model_name)
    print("------------------------------------")
    print("Evaluate the model on the training data set:")
    print(classification_report(training_labels ,prediction ))
    print('Accuracy: {:.2f}%'.format(accuracy_score(training_labels,prediction)*100) )
    print()

    #Evaluate the model on the testing data set
    prediction = model.predict(testing_data)
    print("Evaluate the model on the testing data set:")
    print(classification_report(testing_labels ,prediction ))
    print('Accuracy: {:.2f}%'.format(accuracy_score(testing_labels,prediction)*100) )
    print()

def test_model(model):
    model.fit(training_data, training_labels)
    prediction = model.predict(testing_data)
    score = accuracy_score(testing_labels,prediction)*100

    return score


def find_best_k_for_KNN():
    
    best_k = 1
    best_score = 0
    for k in range(2,30):
        model = KNeighborsClassifier(n_neighbors = k)
        score = test_model(model)

        if score > best_score:
            best_k = k
            best_score = score

    return best_k

def find_best_model(models):
    best_score = 0
    model_name = 0

    print("\nModel : score")
    print("--------------------------")
    for key in models:
        model = models[key]

        if key == 'Knn':
            k = find_best_k_for_KNN()
            model = KNeighborsClassifier(n_neighbors = k)
            
        score = test_model(model)
        print(key, ": {:.2f}%".format(score))

        if score > best_score:
            best_score = score
            model_name = key

    return model_name, best_score


#text processing and cleaning function
def process_text(text):
    
    #1 Remove Punctuationa
    no_punctuations = [character for character in text if character not in string.punctuation]
    no_punctuations = ''.join(no_punctuations)
    
    #2 Remove Stop Words, and lower case
    no_stopwords = [word.lower() for word in no_punctuations.split() if word.lower() not in stopwords.words('english')]
    
    # 3 stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in no_stopwords]
    
    # 4 delete numbers, and 1 letter words
    for index, word in enumerate(stemmed):
        if word.isalpha() == False: 
            del stemmed[index]
            
        elif len(word) == 1:
            del stemmed[index]
    
    return stemmed


########################################################################################################################
########################################################################################################################

#Load the data
print("Loading the data from text file...")
emails_data = pandas.read_csv('emails.csv')

#Checking for duplicates and removing them
emails_data.drop_duplicates(inplace = True)

#text processing, and tokenization with their counts
print("Preprocessing the text...")
processed_emails_data = CountVectorizer(analyzer=process_text).fit_transform(emails_data['text'])

#Split data into 75% training & 25% testing data sets
print("training the classifier...")
training_data, testing_data, training_labels, testing_labels = train_test_split(processed_emails_data, emails_data['spam'], test_size = 0.25, random_state = 42)

#train on multiple models and compare them
models = {
  "Naive Bayes": MultinomialNB(),
  # "Linear SVM 1":  LinearSVC(verbose=0),
  "Linear SVM 2":  SVC(kernel = 'linear', random_state = 42),
  "Knn":         KNeighborsClassifier()
}

print("Testing the classifiers...")
model_name, score = find_best_model(models)

print("\nDisplay best classifier report...")
print_classification_report(model_name, models[model_name])
