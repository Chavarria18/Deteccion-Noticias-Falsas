
from calendar import c
import streamlit as st 
import nltk
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from streamlit_metrics import metric, metric_row
import altair as alt
def machine_analysis(cuerpo):
   
    df= pd.read_csv('./data-final.csv',sep=';',encoding= 'unicode_escape')
    print(df.shape)  



    df.loc[(df['Category'] == 0) , ['Category']] = 'Fake'
    df.loc[(df['Category'] == 1) , ['Category']] = 'True'

   ### print(df.head())

    labels = df.Category

    ###dividir el conjunto de datos en dos sets uno para entrenamiento y otro para probar/test
    x_train,x_test,y_train,y_test=train_test_split(df['Text'].values.astype('str'), labels, test_size=0.20,train_size=0.80, random_state=0)

    ###Quitar las stopwords 
    nltk.download('stopwords')
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words("spanish"), max_df=0.7)

    ###se transforma en un conjutno de entrenamiento y pruebas
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    tfidf_test=tfidf_vectorizer.transform(x_test)

    #se inicializa PassiveAggressiveClassifier 
    pa_classifier=PassiveAggressiveClassifier(max_iter=50)
    pa_classifier.fit(tfidf_train,y_train)

    #Se inicializa el RandomForestClassifier
    randomforest = RandomForestClassifier(max_depth=2, random_state=0)
    randomforest.fit(tfidf_train,y_train)
    #Se inicaliza el SVM classifier
    svmclasf = svm.SVC(probability=True)
    svmclasf.fit(tfidf_train,y_train)


    #####----------------
    ###Acurracy Score
    #####--------------- 
    ###Se realizan las pruebas de los clasificadores
    ###--->Passive agressive 
    y_pred=pa_classifier.predict(tfidf_test)
    score=accuracy_score(y_test,y_pred)
    
    ###--->Random Forest
    y_pred2=randomforest.predict(tfidf_test)
    score2=accuracy_score(y_test,y_pred2)
    ###--->SVM
    y_pred3=svmclasf.predict(tfidf_test)
    score3=accuracy_score(y_test,y_pred3)
    
   
    texto = [cuerpo]
    texto2 = tfidf_vectorizer.transform(texto)

    st.title("Resultados de los clasificadores")
    st.write("Probabilidades %: ")
    st.write("**Support Vector Machine**") 
    col1, col2 = st.columns(2)
    col1.metric("Falsa", str( round( svmclasf.predict_proba(texto2)[:,0][0] ,2)*100) +"%")
    col2.metric("verdadera",  str( round(svmclasf.predict_proba(texto2)[:,1][0] ,2)*100) +"%")  
    


    #st.write(svmclasf.classes_)
    #st.write(svmclasf.predict_proba(texto2))

    #st.write(svmclasf.predict(texto2))

    ##st.write(pa_classifier.predict_proba(texto2))
    # 
    def predict_proba_PA(X):      
        prob = pa_classifier.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob    

    st.write("**Passive Aggresive**")
    col1, col2 = st.columns(2)
    col1.metric("Falsa", str( round(predict_proba_PA(texto2)[:,0][0],2)*100) +"%")
    col2.metric("verdadera", str( round(predict_proba_PA(texto2)[:,1][0],2)*100) +"%")
    
    
    
    st.write("**Random Forest**")
    col1, col2 = st.columns(2)
    col1.metric("Falsa", str( round(randomforest.predict_proba(texto2)[:,0][0],2)*100) +"%")
    col2.metric("verdadera", str( round(randomforest.predict_proba(texto2)[:,1][0],2)*100) +"%")
   

    #st.write(randomforest.predict(texto2))

    


    
   

   







