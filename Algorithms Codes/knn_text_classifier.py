import os
import re
import string
import math
from decimal import Decimal
import pandas as pd

dataset = pd.read_excel('C://Users//Owner//Downloads//lable_tweets (2).xlsx')
dataset=dataset.dropna()
dataset=dataset.reset_index(drop=True)


x=dataset.iloc[:,0]
y=dataset.iloc[:,1]
X=x.to_dict()

X=[]
for d in range(len(x)):
    b=x[d]
    X.append(b)
   
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect=CountVectorizer()
a=count_vect.fit_transform(X)
a.toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.toarray()

from imblearn.over_sampling import SMOTE
sm=SMOTE()
X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, y_train)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2)
clf.fit(X_train_res, y_train_res)


X_test_tfidf=count_vect.transform(X_test)

y_pred=clf.predict(X_test_tfidf)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)

print(cm)
print(Accuracy_Score)
