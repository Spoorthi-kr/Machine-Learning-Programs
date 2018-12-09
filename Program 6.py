from sklearn.datasets import fetch_20newsgroups as ft

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train=ft(subset='train',categories=categories,shuffle=True)
twenty_test=ft(subset='test',categories=categories,shuffle=True)

count_vect=CountVectorizer()        #creating objects
tfidf_transform=TfidfTransformer()
mul=MultinomialNB()

x_train=count_vect.fit_transform(twenty_train.data)
tfidf_train=tfidf_transform.fit_transform(x_train)
mul.fit(tfidf_train,twenty_train.target)  #supervised learning we will take target while learning

x_test=count_vect.transform(twenty_test.data)
tfidf_test=tfidf_transform.transform(x_test)
predicted=mul.predict(tfidf_test)

print("accuracy score is :",accuracy_score(predicted,twenty_test.target))
print("classification report is :",classification_report(predicted,twenty_test.target,target_names=twenty_test.target_names))
print("confusion matrix is :",confusion_matrix(predicted,twenty_test.target))
