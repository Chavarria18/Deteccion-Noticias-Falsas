import streamlit as st 

import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


###Pagina web 
st.title('Streamlit Example')
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Passive Aggresive', 'SVM', 'Random Forest')
)

####Modelo 
df= pd.read_csv('train2.csv')
print(df.shape)




df.loc[(df['label'] == 1) , ['label']] = 'FAKE'
df.loc[(df['label'] == 0) , ['label']] = 'REAL'

print(df.head())

labels = df.label

###dividir el conjunto de datos en dos sets uno para entrenamiento y otro para probar/test
x_train,x_test,y_train,y_test=train_test_split(df['text'].values.astype('str'), labels, test_size=0.001,train_size=0.001, random_state=0)

###Quitar las stopwords 
tfidf_vectorizer = TfidfVectorizer(stop_words='spanish', max_df=0.7)

###se transforma en un conjutno de entrenamiento y pruebas
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#se inicializa PassiveAggressiveClassifier 
pa_classifier=PassiveAggressiveClassifier(max_iter=50)
pa_classifier.fit(tfidf_train,y_train)

###ya se trata de hacer la clasificación  
y_pred=pa_classifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
###encontrar el numero de exitos y fallas para hacerlo 
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])